# monkey patching for SAM change consistency
import segment_anything.modeling.image_encoder
from .new_image_encoder import newImageEncoderViT
segment_anything.modeling.image_encoder.ImageEncoderViT = newImageEncoderViT

import segment_anything.predictor
from .new_sam_predictor import newSamPredictor
segment_anything.predictor.SamPredictor = newSamPredictor

# imports
from .datasets import FTW
from .datamodules import preprocess
from .image_mlp import PixelMLP
from .metrics import get_object_level_metrics, get_object_level_metrics_mpred
from torch.utils.data import DataLoader
from segment_anything import SamPredictor
from .new_automatic_mask_generator import SamAutomaticMaskGenerator
from .build_sam import sam_model_registry
from segment_anything.modeling.image_encoder import PatchEmbed
from segment_anything.modeling.mask_decoder import MaskDecoder
from .sam_mask_decoder_change import new_predict_masks
from .sam_predictor_set_image_change import new_set_image
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch import nn

from ...detections import Detections
from ...converters import convert_sam_output
from ...intermediate_formats import InstanceOutput

import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import os

class SAMSetup:
    def __init__(self, config):
        self.config = config
        self.debug = bool(config.get('debug', {}).get('verbose', False))
        if self.debug:
            print("[SAMSetup] __init__ start", flush=True)
        # device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        if self.debug:
            print(f"[SAMSetup] device={self.device}", flush=True)

        if self.debug:
            print("[SAMSetup] _setup_data...", flush=True)
        self._setup_data(config)

        if self.debug:
            print("[SAMSetup] _setup_model...", flush=True)
        self._setup_model(config)
        if 'img_proj' in config:
            if self.debug:
                print("[SAMSetup] _setup_img_proj...", flush=True)
            self._setup_img_proj(config)
        if 'emb_cond' in config:
            if self.debug:
                print("[SAMSetup] _setup_emb_cond...", flush=True)
            self._setup_emb_cond(config)
        if self.debug:
            print("[SAMSetup] __init__ done", flush=True)
        
        #self.state_train = False
        #self.state_eval = False
        #self._setup_training(config)

    def _setup_data(self, config):
        self.test_countries = config['data']['test_countries']
        self.data_dir = config['data']['data_dir']
        self.load_boundaries = config['data']['load_boundaries']
        self.temporal_options = config['data']['temporal_options']
        self.batch_size = config['data']['batch_size']
        self.use_rgb = config['data']['use_rgb']
        if getattr(self, 'debug', False):
            print(f"[SAMSetup] data: countries={self.test_countries}, data_dir={self.data_dir}", flush=True)

    def _setup_model(self, config):
        if getattr(self, 'debug', False):
            print("[SAMSetup] building SAM model...", flush=True)
        # sam model
        self.model_path = config['sam_model']['model_ckpt']
        self.model_in_chans = config['sam_model']['in_chans']
        self.model_adapt_inp = config['sam_model']['adapt_inp']
        self.model_change_chan_num = config['sam_model']['change_chan_num']
        if getattr(self, 'debug', False):
            exists = os.path.exists(self.model_path)
            size = os.path.getsize(self.model_path) if exists else -1
            print(f"[SAMSetup] checkpoint={self.model_path}, exists={exists}, size={size}", flush=True)
        
        self.sam = sam_model_registry["vit_h"](in_chans=self.model_in_chans, checkpoint=self.model_path)
        if getattr(self, 'debug', False):
            print("[SAMSetup] SAM model created", flush=True)
        if self.model_change_chan_num:
            patch_size = self.sam.image_encoder.patch_size
            in_chans = 8 #sam.image_encoder.in_chans
            embed_dim = self.sam.image_encoder.embed_dim
            self.sam.image_encoder.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            if self.debug:
                print("[SAMSetup] patch_embed replaced for 8-channel input", flush=True)

        if self.model_in_chans==8:
            pixel_mean = 2*[123.675, 116.28, 103.53, 123.675]
            pixel_std = 2*[58.395, 57.12, 57.375, 58.395]
            self.sam.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.sam.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
            if self.debug:
                print("[SAMSetup] registered pixel mean/std for 8-ch", flush=True)

        if self.debug:
            print(f"[SAMSetup] Moving model to device={self.device}... (this may take a moment for large models)", flush=True)
            if self.device == 'cuda':
                print(f"[SAMSetup] CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}", flush=True)
                if torch.cuda.is_available():
                    print(f"[SAMSetup] Current CUDA device: {torch.cuda.current_device()}, device name: {torch.cuda.get_device_name()}", flush=True)
        self.sam.to(self.device)
        if self.debug:
            print("[SAMSetup] SAM model moved to device", flush=True)

        self.predictor = SamPredictor(self.sam)
        self.predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(self.predictor.model.mask_decoder, MaskDecoder)
        if self.debug:
            print("[SAMSetup] predictor ready", flush=True)

    def _setup_img_proj(self, config):
        self.use_img_proj = config['img_proj']['use_img_proj']
        self.img_ckpt = config['img_proj']['img_ckpt']
        self.img_model_params = config['img_proj']['img_model_params']

        if self.use_img_proj:
            if self.debug:
                print("[SAMSetup] loading img_proj PixelMLP...", flush=True)
            self.img_proj = PixelMLP(*self.img_model_params) # in_chn, out_chn, hidden_dim, num_layers
            if self.img_ckpt is not False:
                ckpt = torch.load(img_ckpt)
                self.img_proj.load_state_dict(ckpt['state_dict'])
                del ckpt
            self.img_proj.to(self.device)
            if self.debug:
                print("[SAMSetup] img_proj ready", flush=True)

    def _setup_emb_cond(self, config):
        self.use_emb_cond = config['emb_cond']['use_emb_cond']
        self.emb_ckpt = config['emb_cond']['emb_ckpt']
        self.emb_tokens = config['emb_cond']['emb_tokens']

        if self.use_emb_cond:
            if self.debug:
                print("[SAMSetup] setting up emb_cond...", flush=True)
            embed_dim = self.sam.prompt_encoder.embed_dim
            #print(embed_dim) # 256
            self.emb_cond = nn.Embedding(self.emb_tokens, embed_dim)
            if emb_ckpt is not False:
                ckpt = torch.load(self.emb_ckpt)
                self.emb_cond.load_state_dict(ckpt)
                del ckpt
            self.emb_cond.to(self.device)
            if self.debug:
                print("[SAMSetup] emb_cond ready", flush=True)

    def _get_masks(self,fullmask,nsel=1):
        mids = fullmask.flatten().unique() # first in is 0 (maskless)
        mask_inds = mids[torch.randint(mids.shape[0],(nsel,))] # (nsel,); sample with replacement
        masks, points, labels = [], [], []
        for i in range(nsel):
            mid = mask_inds[i]
            mask = (fullmask==mid)
            pts = torch.argwhere(mask)
            sel_pt = pts[torch.randint(pts.shape[0],())].flip(0) # (2,); [y,x] -> [x,y]
            mask = mask * (mid>0) # set all 0s if background
            masks.append(mask)
            points.append(sel_pt)
            labels.append((mid>0).to(torch.uint8).squeeze(-1))
        return torch.stack(masks), torch.stack(points), torch.stack(labels) # (nsel,256,256), (nsel,2), (nsel,)

    # returns: images (b,c,h,w), masks (b,nsel,h,w), points (b,n,2), labels, (b,nsel)
    def _read_batch(self,batch,nsel=1,device='cpu'):
        if self.use_rgb:
            bimage = batch['image'][:,:3].to(device) # sel RGB channels
        else:
            bimage = batch['image'].to(device)
        bfullmask = batch['mask'].to(device)
        bmasks, bpoints, blabels = [], [], []
        batch_size = bimage.shape[0]
        for i in range(batch_size):
            masks, points, labels = self._get_masks(bfullmask[i],nsel=nsel)
            bmasks.append(masks)
            bpoints.append(points)
            blabels.append(labels)
        bmasks = torch.stack(bmasks).to(device)
        bpoints = torch.stack(bpoints).to(device)
        blabels = torch.stack(blabels).to(device)
        return bimage, bmasks, bpoints, blabels # labels: 1=foreground, 0=background


class SAMTrainer(SAMSetup):
    def __init__(self, config):
        super().__init__(config)

        self._setup_data_train(config)
        self._setup_training(config)

    def _setup_data_train(self, config):
        self.train_countries = config['data']['train_countries']
        self.val_countries = config['data']['val_countries']
        self.train_worker_num = config['data']['train_worker_num']
        self.val_worker_num = config['data']['val_worker_num']

        self.train_dataset = FTW(
            root=self.data_dir,
            countries=self.train_countries,
            split='train',
            transforms=preprocess,
            load_boundaries=self.load_boundaries,
            temporal_options=self.temporal_options
        )
        self.val_dataset = FTW(
            root=self.data_dir,
            countries=self.val_countries,
            split='val',
            transforms=preprocess,
            load_boundaries=self.load_boundaries,
            temporal_options=self.temporal_options
        )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_worker_num)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.val_worker_num)

    def _setup_training(self, config):
        self.exp_name = config['trainer']['exp_name']
        self.epochs = config['trainer']['epochs']
        self.save_every = config['trainer']['save_every']
        self.accumulation_steps = config['trainer']['accumulation_steps']
        self.nsel = config['trainer']['nsel']

        self.opt_model = config['trainer']['opt_model']
        self.train_mask_dec = config['trainer']['train_mask_dec']
        self.train_prompt_enc = config['trainer']['train_prompt_enc']
        self.train_image_enc = config['trainer']['train_image_enc']
        self.lr = config['trainer']['lr']
        self.weight_decay = config['trainer']['weight_decay']

        # Set training parameters
        if self.opt_model:
            self.predictor.model.mask_decoder.train(self.train_mask_dec) # enable training of mask decoder
            self.predictor.model.prompt_encoder.train(self.train_prompt_enc) # enable training of prompt encoder
            self.predictor.model.image_encoder.train(self.train_image_enc) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
            if self.model_change_chan_num or self.model_adapt_inp:
                self.predictor.model.image_encoder.patch_embed.train(True)

        # lr=1e-5
        opt_params = []
        if self.opt_model: opt_params += list(self.predictor.model.parameters()); print('opt model')
        if self.use_img_prep: opt_params += list(self.img_proj.parameters()); print('opt img_prep')
        if self.use_emb_cond: opt_params += list(self.emb_cond.parameters()); print('opt emb_cond')
        print('num opt params: ', len(opt_params))
        self.optimizer = torch.optim.AdamW(params=opt_params,lr=self.lr,weight_decay=self.weight_decay) # lr=2e-5
        self.scaler = torch.cuda.amp.GradScaler() # mixed precision

        self.dl_len = len(self.train_dataloader)
        total_steps = -(-self.dl_len // self.accumulation_steps) * self.epochs
        warmup_steps = int(0.1 * total_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self, epoch, **kwargs):
        device = self.device if 'device' not in kwargs else kwargs['device']
        mean_iou = 0 if 'mean_iou' not in kwargs else kwargs['mean_iou']
        total_loss = 0
        with tqdm(self.train_dataloader, unit='batch') as tdl:
            tdl.set_description(f"Epoch {epoch}")
            for itr, batch in enumerate(tdl):
                with torch.cuda.amp.autocast():
                    images, masks, points, labels = self._read_batch(batch,nsel=nsel,device=self.device) #images (b,c,h,w), masks (b,nsel,h,w), points (b,n,2), labels, (b,nsel)
                    b_size = images.shape[0]

                    ## preprocess imgs if desired
                    if self.use_img_proj:
                        images = self.img_proj(images)

                    ## format inputs
                    original_size = tuple(images.shape[-2:])
                    transformed_images = self.predictor.transform.apply_image_torch(images)
                    transformed_images = (transformed_images*256)#.to(torch.uint8)
                    input_size = tuple(transformed_images.shape[-2:])

                    # flatten batch*nsel for batched mask decoding
                    points = points.flatten(0,1) # (b,nsel,2) -> (b*nsel,2)
                    points = self.predictor.transform.apply_coords_torch(points, original_size)

                    ## predict
                    input_images = self.predictor.model.preprocess(transformed_images)
                    features = self.predictor.model.image_encoder(input_images)
                    # expand features along batch by points nsel
                    features = features.unsqueeze(1).expand(-1,nsel,-1,-1,-1).flatten(0,1) # (b,...) -> (b,1,...) -> (b,nsel,...) -> (b*nsel,...)
                    points_inp = (points.unsqueeze(1), labels.flatten().unsqueeze(-1))
                    # Embed prompts ## sparse_embeddings: (b,1+nsel(bad/num_pts_per_single_obj_seg),256(d))
                    sparse_embeddings, dense_embeddings = self.predictor.model.prompt_encoder(points=points_inp,boxes=None,masks=None)

                    if use_emb_cond:
                        cond_emb = self.emb_cond(torch.zeros((sparse_embeddings.shape[0],),dtype=torch.int,device=device)).unsqueeze(1) # (b,1,emb_dim)
                        sparse_embeddings = torch.cat([sparse_embeddings, cond_emb], dim=1)

                    # Predict masks
                    low_res_masks, prd_scores = self.predictor.model.mask_decoder(
                        image_embeddings=features,
                        image_pe=self.predictor.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )
                    # Upscale/downscale the masks to the original image resolution
                    prd_masks = self.predictor.model.postprocess_masks(low_res_masks, input_size, original_size)
                    # separate batch from nsel in prd_masks
                    prd_masks = prd_masks.unflatten(0,(b_size,nsel)) # (b*nsel,...) -> (b,nsel,...)

                    ## loss calculation
                    #gt_mask = masks.expand(-1,3,-1,-1).to(torch.float32) # (b,1,h,w) -> (b,3,h,w); bool -> float
                    gt_mask = masks.unsqueeze(2).expand(-1,-1,3,-1,-1).to(torch.float32) # (b,nsel,h,w) -> (b,nsel,3,h,w)
                    prd_mask = torch.sigmoid(prd_masks) # Turn logit map to probability map

                    # Segmentation Loss calculation
                    gamma = 2 # higher gamma -> stronger focus on larger misclassifications 
                    alpha = 0.5 # weight between background vs foreground classes
                    focal_loss = (-gt_mask * alpha * (1-prd_mask)**gamma  * torch.log(prd_mask+1e-5) - (1-gt_mask) * (1-alpha) * prd_mask**gamma  * torch.log((1-prd_mask)+1e-5)).mean((-2,-1)) # cross entropy loss X --> Focal loss

                    dice_loss = -(2 * (gt_mask * prd_mask).sum((-2,-1)) / (gt_mask.sum((-2,-1)) + prd_mask.sum((-2,-1)) + 1e-4))#.mean()

                    seg_loss = (focal_loss + dice_loss).min(dim=-1).values.mean() # choose 1 of the 3 masks w/ min loss to backprop to enable specialization

                    # Score loss calculation (intersection over union) IOU
                    inter = (gt_mask * (prd_mask>0.5)).sum((-2,-1))
                    iou = inter / (gt_mask.sum((-2,-1)) + (prd_mask>0.5).sum((-2,-1)) - inter)
                    iou[labels.squeeze(-1)==0] = torch.zeros_like(iou[labels.squeeze(-1)==0])
                    iou = torch.where(torch.logical_or(iou.isnan(),iou.isinf()),torch.zeros_like(iou),iou)
                    score_loss = torch.abs(prd_scores - iou.flatten(0,1).detach()).mean()

                    loss = seg_loss + score_loss*0.05  # mix losses
                    if loss.isnan().any(): print(seg_loss, score_loss, loss)
                    total_loss += loss.item()

                    # apply back propogation
                    self.scaler.scale(loss).backward()  # Backpropogate

                    if (itr+1) % self.accumulation_steps == 0 or (itr+1) == self.dl_len:
                        self.scaler.step(self.optimizer)
                        self.scaler.update() # Mix precision
                        self.scheduler.step()
                        self.predictor.model.zero_grad() # empty gradient

                    #if (itr+1)%10000==0:
                    #    torch.save(self.predictor.model.state_dict(), "checkpoints/sam_{}_{}_{}".format(exp_name,epoch,itr)) # save model

                    # Display results

                    if itr==0: mean_iou=0
                    sel_iou = iou.min(dim=-1).values[labels.squeeze(-1)==1]
                    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(sel_iou.cpu().detach().numpy())
                    #print("step)",itr, "Accuracy(IOU)=",mean_iou)
                    tdl.set_postfix(
                        loss=loss.cpu().item(),
                        iou=mean_iou,
                    )

        mean_loss = total_loss / self.dl_len
        metrics = {
            'mean_loss': mean_loss,
            'mean_iou': mean_iou,
        }
        return metrics

    def validate_epoch(self, epoch, kwargs=None):
        device = self.device if 'device' not in kwargs else kwargs['device']
        val_recall = []
        val_iou = []
        with torch.no_grad():
            with tqdm(self.val_dataloader, unit='batch') as vdl:
                for itr, batch in enumerate(vdl):
                    with torch.cuda.amp.autocast():
                        vdl.set_description(f"Val Epoch {epoch}")

                        images, masks, points, labels = self._read_batch(batch,nsel=nsel,use_rgb=use_rgb,device=device) #images (b,c,h,w), masks (b,nsel,h,w), points (b,n,2), labels, (b,nsel)
                        b_size = images.shape[0]

                        ## preprocess imgs if desired
                        if self.use_img_proj:
                            images = self.img_proj(images)

                        # flatten batch*nsel for batched mask decoding
                        points = points.flatten(0,1) # (b,nsel,2) -> (b*nsel,2)

                        ## format inputs
                        original_size = tuple(images.shape[-2:])
                        transformed_images = self.predictor.transform.apply_image_torch(images)
                        transformed_images = (transformed_images*256).to(torch.uint8)
                        input_size = tuple(transformed_images.shape[-2:])

                        points = self.predictor.transform.apply_coords_torch(points, original_size)

                        ## predict
                        input_images = self.predictor.model.preprocess(transformed_images)
                        features = self.predictor.model.image_encoder(input_images)
                        # expand features along batch by points nsel
                        features = features.unsqueeze(1).expand(-1,nsel,-1,-1,-1).flatten(0,1) # (b,...) -> (b,1,...) -> (b,nsel,...) -> (b*nsel,...)
                        points_inp = (points.unsqueeze(1), labels.flatten().unsqueeze(-1))
                        # Embed prompts ## sparse_embeddings: (b,1+nsel(bad/num_pts_per_single_obj_seg),256(d))
                        sparse_embeddings, dense_embeddings = self.predictor.model.prompt_encoder(points=points_inp,boxes=None,masks=None)
                        if use_emb_cond:
                            cond_emb = self.emb_cond(torch.zeros((sparse_embeddings.shape[0],),dtype=torch.int,device=device)).unsqueeze(1) # (b,1,emb_dim)
                            sparse_embeddings = torch.cat([sparse_embeddings, cond_emb], dim=1)
                        # Predict masks
                        low_res_masks, prd_scores = self.predictor.model.mask_decoder(
                            image_embeddings=features,
                            image_pe=self.predictor.model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=True,
                        )
                        # Upscale/downscale the masks to the original image resolution
                        prd_masks = self.predictor.model.postprocess_masks(low_res_masks, input_size, original_size)
                        # separate batch from nsel in prd_masks
                        prd_masks = self.prd_masks.unflatten(0,(b_size,nsel)) # (b*nsel,...) -> (b,nsel,...)

                        ## loss calculation
                        #gt_mask = masks.expand(-1,3,-1,-1).to(torch.float32) # (b,1,h,w) -> (b,3,h,w); bool -> float
                        gt_mask = masks.unsqueeze(2).expand(-1,-1,3,-1,-1).to(torch.float32) # (b,nsel,h,w) -> (b,nsel,3,h,w)
                        prd_mask = torch.sigmoid(prd_masks) # Turn logit map to probability map

                        # get IOU
                        inter = (gt_mask * (prd_mask>0.5)).sum((-2,-1))
                        iou = inter / (gt_mask.sum((-2,-1)) + (prd_mask>0.5).sum((-2,-1)) - inter)
                        iou[labels.squeeze(-1)==0] = torch.zeros_like(iou[labels.squeeze(-1)==0])
                        iou = torch.where(torch.logical_or(iou.isnan(),iou.isinf()),torch.zeros_like(iou),iou)
                        #score_loss = torch.abs(prd_scores - iou.flatten(0,1).detach()).mean()
                        sel_iou = iou.min(-1).values[labels.squeeze(-1)==1]
                        pseudo_obj_recall = (sel_iou>.5).to(torch.float16).mean().cpu().item()
                        if labels.sum().item()>0:
                            val_recall.append(pseudo_obj_recall)
                            val_iou.append(sel_iou.mean().cpu().item())

                        # Display results
                        vdl.set_postfix(
                            iou=np.array(val_iou).mean(), #sel_iou.mean().cpu().item(),
                            obj_recall=np.array(val_recall).mean(), #pseudo_obj_recall,
                        )

        obj_recall = np.array(val_recall).mean()
        obj_iou = np.array(val_iou).mean()
        metrics = {
            'obj_recall': obj_recall,
            'obj_iou': obj_iou,
        }
        return metrics

    def train(self):
        metrics = {
            'mean_iou': 0,
            'best_val_obj_recall': -torch.inf,
        }
        for epoch in range(self.epochs):
            tr_metrics = self.train_epoch(epoch, mean_iou=metrics['mean_iou'])
            val_metrics = self.validate_epoch(epoch)

            # opt save every
            if self.save_every is not False:
                if (epoch+1)%self.save_every == 0:
                    torch.save(
                        self.predictor.model.state_dict(),
                        "checkpoints/sam_{}_{}_val{:.4f}".format(self.exp_name,epoch,val_metrics['obj_recall'])
                    ) # save model

            # save best
            if val_metrics['obj_recall'] > metrics['best_val_obj_recall']:
                torch.save(
                    self.predictor.model.state_dict(),
                    "checkpoints/sam_final_{}_{}".format(self.exp_name,epoch)
                ) # save model
                if self.use_img_prep:
                    ckpt = {
                        'state_dict': self.img_proj.state_dict(),
                    }
                    torch.save(ckpt, "checkpoints/img_proj_{}_{}".format(self.exp_name,epoch))
                if self.use_emb_cond:
                    torch.save(self.emb_cond.state_dict(), "checkpoints/emb_cond_{}_{}".format(self.exp_name,epoch))

            # save val metrics across exp
            with open('results/{}_vals.csv'.format(self.exp_name), 'a') as f:
                f.write('{}, '.format(val_metrics['obj_recall']))

            metrics['mean_iou'] = tr_metrics['mean_iou']

        print('done training')


class SAMEval(SAMSetup):
    def __init__(self, config):
        if bool(config.get('debug', {}).get('verbose', False)):
            print("[SAMEval] __init__ start", flush=True)
        super().__init__(config)
        if bool(config.get('debug', {}).get('verbose', False)):
            print("[SAMEval] calling _setup_eval...", flush=True)
        self._setup_eval(config)
        if bool(config.get('debug', {}).get('verbose', False)):
            print("[SAMEval] __init__ done", flush=True)

    def _setup_data_eval(self, config):
        if getattr(self, 'debug', False):
            print("[SAMEval] _setup_data_eval start", flush=True)
        self.test_countries = config['data']['test_countries']
        self.test_worker_num = config['data']['test_worker_num']

        self.test_sort_countries = config['eval']['sort_countries']
        order = sorted if self.test_sort_countries else lambda x: x
        if getattr(self, 'debug', False):
            print(f"[SAMEval] countries={self.test_countries}, sort={self.test_sort_countries}", flush=True)

        self.test_dataset_generator = (
            FTW(
                root=self.data_dir,
                countries=[country],
                split='test',
                transforms=preprocess,
                load_boundaries=self.load_boundaries,
                temporal_options=self.temporal_options
            ) for country in order(self.test_countries)
        )

        self.test_dataloader_generator = (
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.test_worker_num
            ) for ds in self.test_dataset_generator
        )
        if getattr(self, 'debug', False):
            print("[SAMEval] _setup_data_eval done", flush=True)

    def _setup_eval(self, config):
        if getattr(self, 'debug', False):
            print("[SAMEval] _setup_eval start", flush=True)
        # params
        self.exp_name = config['eval']['exp_name']
        output_dir = config['eval'].get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)
        self.csv_name = os.path.join(output_dir, self.exp_name + "_results.csv")
        if getattr(self, 'debug', False):
            print(f"[SAMEval] csv path: {self.csv_name}", flush=True)

        # mask generator hyperparams
        self.iou_thresh = config['eval']['iou_thresh']
        self.stability_thresh = config['eval']['stability_thresh']

        self.points_per_side = config['eval']['points_per_side']
        self.points_per_batch = config['eval']['points_per_batch']

        self.box_nms_thresh = config['eval']['box_nms_thresh']
        self.crop_n_layers = config['eval']['crop_n_layers']
        self.crop_nms_thresh = config['eval']['crop_nms_thresh']
        self.crop_overlap_ratio = config['eval']['crop_overlap_ratio']
        self.min_mask_region_area = config['eval']['min_mask_region_area']
        if getattr(self, 'debug', False):
            print("[SAMEval] creating mask_generator...", flush=True)

        # misc params
        self.print_metrics = config['eval']['print_metrics']
        self.save_metrics = config['eval']['save_metrics']

        ## create csv
        if self.save_metrics:
            with open(self.csv_name, "w") as f:
                f.write("Country,IOU,Pixel Precision,Pixel Recall,Obj Precision,Obj Recall\n")
            if getattr(self, 'debug', False):
                print("[SAMEval] CSV header written", flush=True)

        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            pred_iou_thresh=self.iou_thresh,
            stability_score_thresh=self.stability_thresh,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            box_nms_thresh=self.box_nms_thresh,
            crop_n_layers=self.crop_n_layers,
            crop_nms_thresh=self.crop_nms_thresh,
            crop_overlap_ratio=self.crop_overlap_ratio,
            min_mask_region_area=self.min_mask_region_area,
        )
        self.mask_generator.predictor.model.mask_decoder.predict_masks = new_predict_masks.__get__(self.mask_generator.predictor.model.mask_decoder, MaskDecoder)
        self.mask_generator.predictor.set_image = new_set_image.__get__(self.mask_generator.predictor, SamPredictor)
        if getattr(self, 'debug', False):
            print("[SAMEval] mask_generator ready", flush=True)

    @torch.no_grad()
    def run_country(self, dataloader, **kwargs):
        device = self.device if 'device' not in kwargs else kwargs['device']
        detections_list = []

        country = dataloader.dataset.countries[0]
        image_idx = 0
        with tqdm(dataloader, unit='batch') as tdl:
            tdl.set_description(f"Country {country}")
            for itr, batch in enumerate(tdl):
                for ind in range(batch['image'].shape[0]):
                    #image = batch['image'][ind,:in_chans].moveaxis(0,-1) # sel RGB channels
                    #image = (image*256).to(torch.uint8).numpy()
                    if self.use_img_proj:
                        image = batch['image'][ind,:].to(device).unsqueeze(0)
                        image = self.img_proj(image)
                    else:
                        image = batch['image'][ind,:self.model_in_chans].to(device).unsqueeze(0)
                    gt_mask = batch['mask'][ind].to(device) > 0

                    outputs = self.mask_generator.generate(image)
                    #print(outputs)

                    # Use unified pipeline: convert_sam_output → InstanceOutput → Detections
                    instance_output = convert_sam_output(outputs, image_id=image_idx)
                    detections = instance_output.to_detections(
                        min_area=0,  # No filtering at this stage
                        score_threshold=0.0
                    )
                    detections_list.append(detections)
                    image_idx += 1

        return country, detections_list


    @torch.no_grad()
    def eval_country(self, dataloader, **kwargs):
        device = self.device if 'device' not in kwargs else kwargs['device']
        b_iou, b_pxl_prec, b_pxl_recall, b_obj_prec, b_obj_recall = [], [], [], [], []

        country = dataloader.dataset.countries[0]
        with tqdm(dataloader, unit='batch') as tdl:
            tdl.set_description(f"Country {country}")
            for itr, batch in enumerate(tdl):
                for ind in range(batch['image'].shape[0]):
                    #image = batch['image'][ind,:in_chans].moveaxis(0,-1) # sel RGB channels
                    #image = (image*256).to(torch.uint8).numpy()
                    if self.use_img_proj:
                        image = batch['image'][ind,:].to(device).unsqueeze(0)
                        image = self.img_proj(image)
                    else:
                        image = batch['image'][ind,:self.model_in_chans].to(device).unsqueeze(0)
                    gt_mask = batch['mask'][ind].to(device) > 0

                    outputs = self.mask_generator.generate(image)
                    #print(outputs)

                    masks = [torch.tensor(out['segmentation'],device=device) for out in outputs]
                    #print(len(masks))
                    if len(masks)==0: masks = [torch.zeros((gt_mask.shape[-2],gt_mask.shape[-1]),device=device),]

                    pred_mask = torch.stack(masks).sum(0) > 0 # 2-class

                    # pixel metrics
                    inter = torch.logical_and(gt_mask, pred_mask).sum()
                    union = gt_mask.sum() + pred_mask.sum() - inter
                    iou = (inter / union).cpu().item() if union>0 else 1.
                    #print(inter, union, iou)

                    pxl_tps = inter.cpu().item()
                    pxl_fps = torch.logical_and(gt_mask==False, pred_mask).sum().cpu().item()
                    pxl_fns = torch.logical_and(gt_mask, pred_mask==False).sum().cpu().item()

                    pxl_prec = pxl_tps / (pxl_tps + pxl_fps) if pxl_tps>0 or pxl_fps>0 else 0
                    pxl_recall = pxl_tps / (pxl_tps + pxl_fns) if pxl_tps>0 or pxl_fns>0 else 0

                    # object metrics
                    obj_tps, obj_fps, obj_fns = get_object_level_metrics(gt_mask.cpu().to(torch.float32).numpy(), pred_mask.cpu().to(torch.float32).numpy(), iou_threshold=0.5)
                    #o_masks = [mask.cpu().to(torch.float32).numpy() for mask in masks]
                    #obj_tps, obj_fps, obj_fns = get_object_level_metrics_mpred(gt_mask.cpu().to(torch.float32).numpy(), o_masks, iou_threshold=0.5)

                    obj_prec = obj_tps / (obj_tps + obj_fps) if obj_tps>0 or obj_fps>0 else 0
                    obj_recall = obj_tps / (obj_tps + obj_fns) if obj_tps>0 or obj_fns>0 else 0

                    # append metrics
                    b_iou.append(iou)
                    b_pxl_prec.append(pxl_prec); b_pxl_recall.append(pxl_recall)
                    b_obj_prec.append(obj_prec); b_obj_recall.append(obj_recall)

        ## aggregate metrics
        iou = np.array(b_iou).mean()
        pxl_prec = np.array(b_pxl_prec).mean(); pxl_recall = np.array(b_pxl_recall).mean()
        obj_prec = np.array(b_obj_prec).mean(); obj_recall = np.array(b_obj_recall).mean()

        ## print metrics
        if self.print_metrics:
            print(
                "Country {}: IOU {},\npxl precision {}, pxl recall {}, pxl f1 {},\nobj precision {}, obj recall {}, obj f1 {}".format(
                    country,
                    iou, pxl_prec, pxl_recall, 2*pxl_prec*pxl_recall/(pxl_prec+pxl_recall),
                    obj_prec, obj_recall, 2*obj_prec*obj_recall/(obj_prec+obj_recall)
                )
            )

        ## save metrics
        if self.save_metrics:
            with open(self.csv_name, "a") as f:
                f.write("{},{},{},{},{},{}\n".format(country,iou,pxl_prec,pxl_recall,obj_prec,obj_recall))

        metrics = {
            'iou': iou,
            'pxl_prec': pxl_prec,
            'pxl_recall': pxl_recall,
            'obj_prec': obj_prec,
            'obj_recall': obj_recall,
        }
        return country, metrics

    @torch.no_grad()
    def get_detections(self):
        if getattr(self, 'debug', False):
            print("[SAMEval] get_detections: setting up eval data...", flush=True)
        self._setup_data_eval(self.config) # re-run per eval to recreate dl generator
        country_detections = []
        for dataloader in self.test_dataloader_generator:
            if getattr(self, 'debug', False):
                print("[SAMEval] running country...", flush=True)
            country, detections_list = self.run_country(dataloader)
            country_detections.append((country, detections_list))
            del dataloader
        if getattr(self, 'debug', False):
            print("[SAMEval] get_detections done", flush=True)

        return country_detections

    @torch.no_grad()
    def eval(self):
        self._setup_data_eval(self.config) # re-run per eval to recreate dl generator
        iou, pxl_prec, pxl_recall, obj_prec, obj_recall = [], [], [], [], []
        for dataloader in self.test_dataloader_generator:
            country, metrics = self.eval_country(dataloader)
            iou.append(metrics['iou'])
            pxl_prec.append(metrics['pxl_prec'])
            pxl_recall.append(metrics['pxl_recall'])
            obj_prec.append(metrics['obj_prec'])
            obj_recall.append(metrics['obj_recall'])
            del dataloader

        iou = np.array(iou).mean()
        pxl_prec = np.array(pxl_prec).mean(); pxl_recall = np.array(pxl_recall).mean()
        obj_prec = np.array(obj_prec).mean(); obj_recall = np.array(obj_recall).mean()

        ## save metrics
        if self.save_metrics:
            with open(self.csv_name, "a") as f:
                f.write("{},{},{},{},{},{}\n".format('average',iou,pxl_prec,pxl_recall,obj_prec,obj_recall))

        print('done evaluating!')


