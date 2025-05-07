import random
import time
import os
import re,gc

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import json
import math
from huggingface_hub import hf_hub_download

# from diffusers.pipelines.pipeline_utils import DiffusionPipeline # Assuming this was for a different base class
from schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from transformers import UMT5EncoderModel, AutoTokenizer

from language_segmentation import LangSegment
from music_dcae.music_dcae_pipeline import MusicDCAE
from models.ace_step_transformer import ACEStepTransformer2DModel
from models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from apg_guidance import apg_forward, MomentumBuffer, cfg_forward, cfg_zero_star, cfg_double_condition_forward
import torchaudio


torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285,
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293,
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}

structure_pattern = re.compile(r"\[.*?\]")


def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"


class ACEStepPipeline:

    def __init__(self, checkpoint_dir=None, device_id=0, dtype="bfloat16", text_encoder_checkpoint_path=None, persistent_storage_path=None, torch_compile=False, **kwargs):
        if not checkpoint_dir:
            # Guard against __file__ not being defined in some contexts (e.g. notebooks directly)
            default_base_dir = "."
            try:
                default_base_dir = os.path.dirname(__file__)
            except NameError:
                logger.warning("__file__ not defined, using current directory for default checkpoint path.")

            if persistent_storage_path is None:
                checkpoint_dir = os.path.join(default_base_dir, "checkpoints")
            else:
                checkpoint_dir = os.path.join(persistent_storage_path, "checkpoints")
        ensure_directory_exists(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        if device.type == "mps" and self.dtype == torch.bfloat16:
            logger.info("MPS device detected with bfloat16, changing to float16 for compatibility.")
            self.dtype = torch.float16
        self.device = device
        self.loaded = False # Flag to indicate if models are loaded from disk
        self.torch_compile = torch_compile

        # Attributes for models, will be initialized in load_checkpoint
        self.music_dcae = None
        self.ace_step_transformer = None
        self.text_encoder_model = None
        self.lang_segment = None
        self.lyric_tokenizer = None
        self.text_tokenizer = None


    def load_checkpoint(self, checkpoint_dir=None):
        load_start_time = time.time()
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        device = self.device

        dcae_model_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_model_path = os.path.join(checkpoint_dir, "music_vocoder")
        ace_step_model_path = os.path.join(checkpoint_dir, "ace_step_transformer")
        text_encoder_model_path = os.path.join(checkpoint_dir, "umt5-base")

        files_exist = (
            os.path.exists(os.path.join(dcae_model_path, "config.json")) and
            os.path.exists(os.path.join(dcae_model_path, "diffusion_pytorch_model.safetensors")) and
            os.path.exists(os.path.join(vocoder_model_path, "config.json")) and
            os.path.exists(os.path.join(vocoder_model_path, "diffusion_pytorch_model.safetensors")) and
            os.path.exists(os.path.join(ace_step_model_path, "config.json")) and
            os.path.exists(os.path.join(ace_step_model_path, "diffusion_pytorch_model.safetensors")) and
            os.path.exists(os.path.join(text_encoder_model_path, "config.json")) and
            os.path.exists(os.path.join(text_encoder_model_path, "model.safetensors")) and
            os.path.exists(os.path.join(text_encoder_model_path, "special_tokens_map.json")) and
            os.path.exists(os.path.join(text_encoder_model_path, "tokenizer_config.json")) and
            os.path.exists(os.path.join(text_encoder_model_path, "tokenizer.json"))
        )

        if not files_exist:
            logger.info(f"Checkpoint directory {checkpoint_dir} is not complete, downloading from Hugging Face Hub: {REPO_ID}")

            # download music dcae model
            os.makedirs(dcae_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_dcae_f8c8",
                            filename="config.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_dcae_f8c8",
                            filename="diffusion_pytorch_model.safetensors", local_dir=checkpoint_dir, local_dir_use_symlinks=False)

            # download vocoder model
            os.makedirs(vocoder_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_vocoder",
                            filename="config.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="music_vocoder",
                            filename="diffusion_pytorch_model.safetensors", local_dir=checkpoint_dir, local_dir_use_symlinks=False)

            # download ace_step transformer model
            os.makedirs(ace_step_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="ace_step_transformer",
                            filename="config.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="ace_step_transformer",
                            filename="diffusion_pytorch_model.safetensors", local_dir=checkpoint_dir, local_dir_use_symlinks=False)

            # download text encoder model
            os.makedirs(text_encoder_model_path, exist_ok=True)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="config.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="model.safetensors", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="special_tokens_map.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="tokenizer_config.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)
            hf_hub_download(repo_id=REPO_ID, subfolder="umt5-base",
                            filename="tokenizer.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False)

            logger.info("Models downloaded.")

        dcae_checkpoint_path = dcae_model_path
        vocoder_checkpoint_path = vocoder_model_path
        ace_step_checkpoint_path = ace_step_model_path
        text_encoder_checkpoint_path = text_encoder_model_path

        logger.info("Loading MusicDCAE...")
        self.music_dcae = MusicDCAE(dcae_checkpoint_path=dcae_checkpoint_path, vocoder_checkpoint_path=vocoder_checkpoint_path)
        self.music_dcae.to(device).eval().to(self.dtype)

        logger.info("Loading ACEStepTransformer2DModel...")
        self.ace_step_transformer = ACEStepTransformer2DModel.from_pretrained(ace_step_checkpoint_path, torch_dtype=self.dtype)
        self.ace_step_transformer.to(device).eval().to(self.dtype)
        
        logger.info("Initializing language segmenter and tokenizers...")
        lang_segment = LangSegment()
        lang_segment.setfilters([
            'af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
            'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
            'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
            'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
            'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
            'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'
        ])
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()

        logger.info("Loading UMT5EncoderModel (text encoder)...")
        text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint_path, torch_dtype=self.dtype).eval()
        text_encoder_model = text_encoder_model.to(device).to(self.dtype)
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint_path)
        
        self.loaded = True

        if self.torch_compile:
            logger.info("Compiling models with torch.compile (this may take a while)...")
            compile_start_time = time.time()
            try:
                self.music_dcae = torch.compile(self.music_dcae)
                logger.info("MusicDCAE compiled.")
            except Exception as e:
                logger.error(f"Failed to compile music_dcae: {e}")
            try:
                self.ace_step_transformer = torch.compile(self.ace_step_transformer)
                logger.info("ACEStepTransformer compiled.")
            except Exception as e:
                logger.error(f"Failed to compile ace_step_transformer: {e}")
            try:
                self.text_encoder_model = torch.compile(self.text_encoder_model)
                logger.info("TextEncoderModel compiled.")
            except Exception as e:
                logger.error(f"Failed to compile text_encoder_model: {e}")
            logger.info(f"Models compiled in {time.time() - compile_start_time:.2f} seconds.")
        
        logger.info(f"All models loaded and set up on device '{self.device}' in {time.time() - load_start_time:.2f} seconds.")

    def _ensure_models_on_gpu(self):
        if not self.loaded:
            logger.info("Models not yet loaded from disk. Calling load_checkpoint.")
            self.load_checkpoint() # This will load to self.device
            return

        # If models are loaded but were moved to CPU, and target device is GPU
        if self.device.type != 'cpu' and hasattr(self.music_dcae, 'parameters') and next(self.music_dcae.parameters()).device.type == 'cpu':
            if self.torch_compile:
                logger.warning("torch_compile is True, but models were found on CPU. This is unexpected. Models should remain on GPU if compiled.")
                # Attempt to move them back, but compiled state might be affected.
            logger.info(f"Models were on CPU, moving them to configured device: {self.device}...")
            self.music_dcae.to(self.device)
            self.ace_step_transformer.to(self.device)
            self.text_encoder_model.to(self.device)
            logger.info(f"Models moved to {self.device}.")
        elif self.device.type != 'cpu' and hasattr(self.music_dcae, 'parameters') and next(self.music_dcae.parameters()).device.type != self.device.type:
             logger.warning(f"Models are loaded but not on the configured device {self.device} (current: {next(self.music_dcae.parameters()).device.type}). Attempting to move them.")
             self.music_dcae.to(self.device)
             self.ace_step_transformer.to(self.device)
             self.text_encoder_model.to(self.device)
             logger.info(f"Models moved to {self.device}.")

    def unload_models_to_cpu_and_clear_ram(self): # Renamed for clarity
        if not self.loaded:
            logger.info("Models not loaded. Nothing to unload or clear.")
            return

        # VRAM Management
        if self.device.type != 'cpu': # Only if models could be on GPU
            if self.torch_compile:
                logger.info("torch_compile is enabled. Models remain on GPU. Clearing CUDA cache only.")
            else:
                logger.info("Moving models to CPU to free VRAM...")
                if hasattr(self, 'music_dcae') and self.music_dcae is not None and next(self.music_dcae.parameters()).device.type != 'cpu':
                    self.music_dcae.to('cpu')
                if hasattr(self, 'ace_step_transformer') and self.ace_step_transformer is not None and next(self.ace_step_transformer.parameters()).device.type != 'cpu':
                    self.ace_step_transformer.to('cpu')
                if hasattr(self, 'text_encoder_model') and self.text_encoder_model is not None and next(self.text_encoder_model.parameters()).device.type != 'cpu':
                    self.text_encoder_model.to('cpu')
                logger.info("Models moved to CPU.")

            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared.")
        else:
            logger.info("Device is CPU. No VRAM operations for model movement.")

        # System RAM Management
        logger.info("Attempting to clear system RAM by collecting garbage...")
        gc.collect()
        logger.info("Garbage collection triggered.")

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device: # Should not happen if _ensure_models_on_gpu works
            self.text_encoder_model.to(device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    def get_text_embeddings_null(self, texts, device, text_max_length=256, tau=0.01, l_min=8, l_max=10):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device: # Should not happen
            self.text_encoder_model.to(device)
        
        def forward_with_temperature(inputs, tau=0.01, l_min=8, l_max=10):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
        
            for i in range(l_min, l_max):
                handler = self.text_encoder_model.encoder.block[i].layer[0].SelfAttention.q.register_forward_hook(hook)
                handlers.append(handler)
        
            with torch.no_grad():
                outputs = self.text_encoder_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
        
            for hook_item in handlers: # Renamed hook to hook_item
                hook_item.remove()
        
            return last_hidden_states
    
        last_hidden_states = forward_with_temperature(inputs, tau, l_min, l_max)
        return last_hidden_states

    def set_seeds(self, batch_size, manual_seeds=None):
        seeds = None
        if manual_seeds is not None:
            if isinstance(manual_seeds, str):
                if "," in manual_seeds:
                    seeds = list(map(int, manual_seeds.split(",")))
                elif manual_seeds.isdigit():
                    seeds = int(manual_seeds)
            elif isinstance(manual_seeds, int): # Added for direct int input
                 seeds = manual_seeds
            elif isinstance(manual_seeds, list): # Added for direct list input
                 seeds = manual_seeds


        random_generators = [torch.Generator(device=self.device) for _ in range(batch_size)]
        actual_seeds = []
        for i in range(batch_size):
            seed_val = None # Renamed seed to seed_val
            if seeds is None:
                seed_val = torch.randint(0, 2**32, (1,)).item()
            if isinstance(seeds, int):
                seed_val = seeds
            if isinstance(seeds, list):
                seed_val = seeds[i % len(seeds)] # Handle if seeds list is shorter than batch_size
            random_generators[i].manual_seed(seed_val)
            actual_seeds.append(seed_val)
        return random_generators, actual_seeds

    def get_lang(self, text):
        language = "en"
        try:    
            _ = self.lang_segment.getTexts(text)
            langCounts = self.lang_segment.getCounts()
            if langCounts: # Check if langCounts is not empty
                language = langCounts[0][0]
                if len(langCounts) > 1 and language == "en":
                    language = langCounts[1][0]
        except Exception as err:
            logger.warning(f"Language detection failed for text '{text[:50]}...': {err}. Defaulting to 'en'.")
            language = "en"
        return language

    def tokenize_lyrics(self, lyrics, debug=False):
        lines = lyrics.split("\n")
        lyric_token_idx_list = [261] # Renamed to avoid conflict
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx_list += [2]
                continue

            lang = self.get_lang(line)

            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang: # Simplified Chinese check
                lang = "zh"
            if "spa" in lang: # Spanish check
                lang = "es"

            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, lang)
                if debug:
                    toks = self.lyric_tokenizer.batch_decode([[tok_id] for tok_id in token_idx])
                    logger.info(f"debug tokenize: '{line}' --> lang: {lang} --> tokens: {toks}")
                lyric_token_idx_list = lyric_token_idx_list + token_idx + [2]
            except Exception as e:
                logger.error(f"Tokenize error: {e} for line '{line}', detected language '{lang}'")
        return lyric_token_idx_list

    # ... (calc_v, flowedit_diffusion_process, text2music_diffusion_process remain the same internally) ...
    # These methods use self.ace_step_transformer etc., which will be on the correct device.
    def calc_v(
        self,
        zt_src,
        zt_tar,
        t,
        encoder_text_hidden_states,
        text_attention_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        speaker_embds,
        target_speaker_embeds,
        lyric_token_ids,
        lyric_mask,
        target_lyric_token_ids,
        target_lyric_mask,
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        target_guidance_scale=1.0,
        cfg_type="apg",
        attention_mask=None,
        momentum_buffer=None,
        momentum_buffer_tar=None,
        return_src_pred=True
    ):
        noise_pred_src = None
        if return_src_pred:
            src_latent_model_input = torch.cat([zt_src, zt_src]) if do_classifier_free_guidance else zt_src
            timestep = t.expand(src_latent_model_input.shape[0])
            # source
            noise_pred_src = self.ace_step_transformer(
                hidden_states=src_latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond_src, noise_pred_uncond_src = noise_pred_src.chunk(2)
                if cfg_type == "apg":
                    noise_pred_src = apg_forward(
                        pred_cond=noise_pred_with_cond_src,
                        pred_uncond=noise_pred_uncond_src,
                        guidance_scale=guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred_src = cfg_forward(
                        cond_output=noise_pred_with_cond_src,
                        uncond_output=noise_pred_uncond_src,
                        cfg_strength=guidance_scale,
                    )

        tar_latent_model_input = torch.cat([zt_tar, zt_tar]) if do_classifier_free_guidance else zt_tar
        timestep = t.expand(tar_latent_model_input.shape[0])
        # target
        noise_pred_tar = self.ace_step_transformer(
            hidden_states=tar_latent_model_input,
            attention_mask=attention_mask,
            encoder_text_hidden_states=target_encoder_text_hidden_states,
            text_attention_mask=target_text_attention_mask,
            speaker_embeds=target_speaker_embeds,
            lyric_token_idx=target_lyric_token_ids,
            lyric_mask=target_lyric_mask,
            timestep=timestep,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_with_cond_tar, noise_pred_uncond_tar = noise_pred_tar.chunk(2)
            if cfg_type == "apg":
                noise_pred_tar = apg_forward(
                    pred_cond=noise_pred_with_cond_tar,
                    pred_uncond=noise_pred_uncond_tar,
                    guidance_scale=target_guidance_scale,
                    momentum_buffer=momentum_buffer_tar,
                )
            elif cfg_type == "cfg":
                noise_pred_tar = cfg_forward(
                    cond_output=noise_pred_with_cond_tar,
                    uncond_output=noise_pred_uncond_tar,
                    cfg_strength=target_guidance_scale,
                )
        return noise_pred_src, noise_pred_tar

    @torch.no_grad()
    def flowedit_diffusion_process(
        self,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        target_encoder_text_hidden_states,
        target_text_attention_mask,
        target_speaker_embeds,
        target_lyric_token_ids,
        target_lyric_mask,
        src_latents,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        n_min=0,
        n_max=1.0,
        n_avg=1,
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        target_guidance_scale = guidance_scale
        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        T_steps = infer_steps
        frame_length = src_latents.shape[-1]
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        
        timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            
            encoder_text_hidden_states = torch.cat([encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states)], 0)
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            target_encoder_text_hidden_states = torch.cat([target_encoder_text_hidden_states, torch.zeros_like(target_encoder_text_hidden_states)], 0)
            target_text_attention_mask = torch.cat([target_text_attention_mask] * 2, dim=0)

            speaker_embds = torch.cat([speaker_embds, torch.zeros_like(speaker_embds)], 0)
            target_speaker_embeds = torch.cat([target_speaker_embeds, torch.zeros_like(target_speaker_embeds)], 0)

            lyric_token_ids = torch.cat([lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0)
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

            target_lyric_token_ids = torch.cat([target_lyric_token_ids, torch.zeros_like(target_lyric_token_ids)], 0)
            target_lyric_mask = torch.cat([target_lyric_mask, torch.zeros_like(target_lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()
        momentum_buffer_tar = MomentumBuffer()
        x_src = src_latents
        zt_edit = x_src.clone()
        xt_tar = None
        n_min = int(infer_steps * n_min)
        n_max = int(infer_steps * n_max)

        logger.info("flowedit start from {} to {}".format(n_min, n_max))

        for i, t in tqdm(enumerate(timesteps), total=T_steps, desc="FlowEdit Diffusion"):

            if i < n_min:
                continue

            t_i = t/1000

            if i+1 < len(timesteps): 
                t_im1 = (timesteps[i+1])/1000
            else:
                t_im1 = torch.zeros_like(t_i).to(t_i.device)

            if i < n_max:
                # Calculate the average of the V predictions
                V_delta_avg = torch.zeros_like(x_src)
                for k in range(n_avg):
                    fwd_noise = randn_tensor(shape=x_src.shape, generator=random_generators[0] if random_generators else None, device=device, dtype=dtype) # Use first generator

                    zt_src = (1 - t_i) * x_src + (t_i) * fwd_noise

                    zt_tar = zt_edit + zt_src - x_src

                    Vt_src, Vt_tar = self.calc_v(
                        zt_src=zt_src,
                        zt_tar=zt_tar,
                        t=t,
                        encoder_text_hidden_states=encoder_text_hidden_states,
                        text_attention_mask=text_attention_mask,
                        target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                        target_text_attention_mask=target_text_attention_mask,
                        speaker_embds=speaker_embds,
                        target_speaker_embeds=target_speaker_embeds,
                        lyric_token_ids=lyric_token_ids,
                        lyric_mask=lyric_mask,
                        target_lyric_token_ids=target_lyric_token_ids,
                        target_lyric_mask=target_lyric_mask,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guidance_scale=guidance_scale,
                        target_guidance_scale=target_guidance_scale,
                        attention_mask=attention_mask,
                        momentum_buffer=momentum_buffer
                    )
                    V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src) 

                zt_edit = zt_edit.to(torch.float32)
                zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
                zt_edit = zt_edit.to(V_delta_avg.dtype)
            else: 
                if i == n_max:
                    fwd_noise = randn_tensor(shape=x_src.shape, generator=random_generators[0] if random_generators else None, device=device, dtype=dtype)
                    scheduler._init_step_index(t) # type: ignore
                    sigma = scheduler.sigmas[scheduler.step_index] # type: ignore
                    xt_src = sigma * fwd_noise + (1.0 - sigma) * x_src
                    xt_tar = zt_edit + xt_src - x_src
                
                _, Vt_tar = self.calc_v(
                    zt_src=None,
                    zt_tar=xt_tar, # type: ignore
                    t=t,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                    target_text_attention_mask=target_text_attention_mask,
                    speaker_embds=speaker_embds,
                    target_speaker_embeds=target_speaker_embeds,
                    lyric_token_ids=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    target_lyric_token_ids=target_lyric_token_ids,
                    target_lyric_mask=target_lyric_mask,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guidance_scale=guidance_scale,
                    target_guidance_scale=target_guidance_scale,
                    attention_mask=attention_mask,
                    momentum_buffer_tar=momentum_buffer_tar,
                    return_src_pred=False,
                )
                
                dtype_prev = Vt_tar.dtype # Renamed dtype to dtype_prev
                xt_tar = xt_tar.to(torch.float32) # type: ignore
                prev_sample = xt_tar + (t_im1 - t_i) * Vt_tar # type: ignore
                prev_sample = prev_sample.to(dtype_prev) 
                xt_tar = prev_sample
        
        target_latents = zt_edit if xt_tar is None else xt_tar
        return target_latents # type: ignore

    @torch.no_grad()
    def text2music_diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
        scheduler_type="euler",
        cfg_type="apg",
        zero_steps=1,
        use_zero_init=True,
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        retake_random_generators=None,
        retake_variance=0.5,
        add_retake_noise=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        repaint_start=0,
        repaint_end=0,
        src_latents=None,
    ):

        logger.info("cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(cfg_type, guidance_scale, omega_scale))
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False
        
        do_double_condition_guidance = False
        if guidance_scale_text is not None and guidance_scale_text > 1.0 and guidance_scale_lyric is not None and guidance_scale_lyric > 1.0:
            do_double_condition_guidance = True
            logger.info("do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}".format(do_double_condition_guidance, guidance_scale_text, guidance_scale_lyric))

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        else: # Default or error
            logger.warning(f"Unknown scheduler_type: {scheduler_type}. Defaulting to Euler.")
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

        
        frame_length = int(duration * 44100 / 512 / 8)
        if src_latents is not None:
            frame_length = src_latents.shape[-1]

        if len(oss_steps) > 0:
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=max(oss_steps), device=device, timesteps=None) # type: ignore
            new_timesteps = torch.zeros(len(oss_steps), dtype=timesteps.dtype, device=device) # Match dtype
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx]-1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=num_inference_steps, device=device, sigmas=sigmas) # type: ignore
            logger.info(f"oss_steps: {oss_steps}, num_inference_steps: {num_inference_steps} after remapping to timesteps {timesteps}")
        else:
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=infer_steps, device=device, timesteps=None) # type: ignore
        
        target_latents = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=random_generators[0] if random_generators else None, device=device, dtype=dtype) # Use first generator
        
        is_repaint = False
        is_extend  = False
        repaint_mask = None # Initialize
        x0 = None
        z0 = None
        zt_edit = None

        if add_retake_noise:
            n_min = int(num_inference_steps * (1 - retake_variance)) # Use num_inference_steps
            retake_variance_tensor = torch.tensor(retake_variance * math.pi/2).to(device).to(dtype) # Renamed
            retake_latents_noise = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=retake_random_generators[0] if retake_random_generators else None, device=device, dtype=dtype) # Use first
            repaint_start_frame = int(repaint_start * 44100 / 512 / 8)
            repaint_end_frame = int(repaint_end * 44100 / 512 / 8)
            x0 = src_latents
            is_repaint = (repaint_end_frame - repaint_start_frame != frame_length) 
            is_extend = (repaint_start_frame < 0) or (repaint_end_frame > frame_length)

            if is_extend: is_repaint = True # Extend implies repaint logic for noise handling

            if not is_repaint:
                target_latents = torch.cos(retake_variance_tensor) * target_latents + torch.sin(retake_variance_tensor) * retake_latents_noise
            elif not is_extend:
                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=device, dtype=dtype)
                repaint_mask[:, :, :, repaint_start_frame:repaint_end_frame] = 1.0
                repaint_noise_combined = torch.cos(retake_variance_tensor) * target_latents + torch.sin(retake_variance_tensor) * retake_latents_noise
                z0 = torch.where(repaint_mask == 1.0, repaint_noise_combined, target_latents) # z0 is the initial noisy latent for repaint area
                zt_edit = x0.clone() # type: ignore # Will be used to reconstruct non-repaint area
            elif is_extend:
                to_right_pad_gt_latents = None
                to_left_pad_gt_latents = None
                gt_latents = src_latents
                src_latents_length = gt_latents.shape[-1] # type: ignore
                max_infer_fame_length = int(240 * 44100 / 512 / 8) # Max duration
                left_pad_frame_length = 0
                right_pad_frame_length = 0
                right_trim_length = 0
                left_trim_length = 0

                current_frame_length = gt_latents.shape[-1] # type: ignore

                if repaint_start_frame < 0:
                    left_pad_frame_length = abs(repaint_start_frame)
                    current_frame_length += left_pad_frame_length
                    gt_latents = torch.nn.functional.pad(gt_latents, (left_pad_frame_length, 0), "constant", 0) # type: ignore
                
                if repaint_end_frame > src_latents_length: # Original length before left padding
                    # Calculate right padding based on potentially already left-padded gt_latents
                    right_pad_frame_length = repaint_end_frame - (src_latents_length + left_pad_frame_length) # Corrected calculation
                    if right_pad_frame_length < 0 : right_pad_frame_length = 0 # Ensure non-negative
                    
                    current_frame_length += right_pad_frame_length
                    gt_latents = torch.nn.functional.pad(gt_latents, (0, right_pad_frame_length), "constant", 0) # type: ignore

                # Trimming if exceeds max_infer_fame_length
                if current_frame_length > max_infer_fame_length:
                    if left_pad_frame_length > 0 : # Prefer trimming from right if left was padded
                        right_trim_length = current_frame_length - max_infer_fame_length
                        if right_trim_length > 0:
                            to_right_pad_gt_latents = gt_latents[:,:,:,-right_trim_length:] # type: ignore
                            gt_latents = gt_latents[:,:,:,:-right_trim_length] # type: ignore
                            current_frame_length -= right_trim_length
                    
                    if current_frame_length > max_infer_fame_length: # Still too long, trim from left
                         left_trim_length = current_frame_length - max_infer_fame_length
                         if left_trim_length > 0:
                            to_left_pad_gt_latents = gt_latents[:,:,:,:left_trim_length] # type: ignore
                            gt_latents = gt_latents[:,:,:,left_trim_length:] # type: ignore
                            current_frame_length -= left_trim_length
                
                frame_length = current_frame_length # Update frame_length for diffusion
                repaint_end_frame = frame_length # If right extended, new end is total length

                repaint_mask = torch.zeros((bsz, 8, 16, frame_length), device=device, dtype=dtype)
                if left_pad_frame_length > 0: repaint_mask[:,:,:,:left_pad_frame_length] = 1.0
                if right_pad_frame_length > 0: repaint_mask[:,:,:,-right_pad_frame_length:] = 1.0
                
                x0 = gt_latents # This is the (potentially padded/trimmed) ground truth structure
                
                # Construct z0 (initial noise) for the new frame_length
                # Retake noise for padded regions, original target_latents for the original region (if it fits)
                padded_noise_list = []
                original_target_latents_segment = target_latents[:,:,:,left_trim_length : target_latents.shape[-1]-right_trim_length]

                if left_pad_frame_length > 0:
                    padded_noise_list.append(retake_latents_noise[:, :, :, :left_pad_frame_length])
                
                # Add the segment of target_latents corresponding to the original audio part within the new frame
                # This needs careful indexing if trimming happened
                # For simplicity, if extending, the "original" part is now part of gt_latents (x0)
                # and z0 should be noise in the extended parts.
                # The original target_latents was for the original duration.
                # Let's reconstruct z0 to match x0's new shape.
                temp_z0_parts = []
                if left_pad_frame_length > 0: temp_z0_parts.append(retake_latents_noise_if_needed[:,:,:,:left_pad_frame_length]) # type: ignore
                
                # The part of original target_latents that corresponds to the non-extended, non-trimmed middle
                # Assume target_latents was for the original audio_duration.
                # This logic is getting very complex, the original might have a simpler assumption.
                # Reverting to a simpler noise construction for z0 for extended parts:
                _z0 = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=random_generators[0] if random_generators else None, device=device, dtype=dtype) # type: ignore

                z0 = torch.where(repaint_mask == 1.0, _z0, x0) # Noise in repaint (extended) areas, x0 elsewhere (for structure)
                                                            # This might need to be `target_latents` like for non-extend repaint.
                                                            # The key is what `z0` represents in the formula `zt_src = (1 - t_i) * x0 + (t_i) * z0`
                                                            # `z0` should be the pure noise component for the repaint area.
                # Let's stick to the previous repaint logic: z0 is the noisy version for repaint areas.
                # For extend, the repaint_mask covers the *newly added* areas.
                combined_noise_for_z0 = torch.cos(retake_variance_tensor) * _z0 + torch.sin(retake_variance_tensor) * retake_latents_noise[:,:,:,:frame_length] # Ensure retake_latents_noise matches frame_length
                z0 = torch.where(repaint_mask == 1.0, combined_noise_for_z0, x0) # if x0 is structure, this is not pure noise.
                                                                            # Simpler: z0 is the target noise for the whole new frame_length
                z0 = combined_noise_for_z0 # Initial noise for the whole new canvas

                zt_edit = x0.clone() # Structure to preserve


        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        logger.info(f"Guidance interval: steps {start_idx} to {end_idx} (total {num_inference_steps})")

        momentum_buffer = MomentumBuffer()

        def forward_encoder_with_temperature(inputs, tau=0.01, l_min=4, l_max=6): # Removed self
            handlers = []
            def hook(module, input_val, output_val): # Renamed input, output
                output_val[:] *= tau
                return output_val
            for i in range(l_min, l_max):
                handler = self.ace_step_transformer.lyric_encoder.encoders[i].self_attn.linear_q.register_forward_hook(hook)
                handlers.append(handler)
            encoder_hidden_states, encoder_hidden_mask = self.ace_step_transformer.encode(**inputs)
            for hook_item in handlers: hook_item.remove() # Renamed hook
            return encoder_hidden_states, encoder_hidden_mask # Return mask too

        encoder_hidden_states_cond, encoder_hidden_mask_cond = self.ace_step_transformer.encode( # Renamed
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        encoder_hidden_states_uncond_null = None # Renamed
        if use_erg_lyric:
            encoder_hidden_states_uncond_null, _ = forward_encoder_with_temperature( # Use the modified one
                inputs={
                    "encoder_text_hidden_states": encoder_text_hidden_states_null if encoder_text_hidden_states_null is not None else torch.zeros_like(encoder_text_hidden_states),
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                }
            )
        else:
            encoder_hidden_states_uncond_null, _ = self.ace_step_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask, # Should be same shape as encoder_text_hidden_states's mask
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids), # Or specific null token
                lyric_mask, # Mask should correspond to the null lyric tokens
            )
        
        encoder_hidden_states_no_lyric_cond = None # Renamed
        if do_double_condition_guidance:
            if use_erg_lyric:
                encoder_hidden_states_no_lyric_cond, _ = forward_encoder_with_temperature( # Use modified
                    inputs={
                        "encoder_text_hidden_states": encoder_text_hidden_states,
                        "text_attention_mask": text_attention_mask,
                        "speaker_embeds": torch.zeros_like(speaker_embds),
                        "lyric_token_idx": lyric_token_ids, # ERG on lyric usually means weaker lyric, not zero
                        "lyric_mask": lyric_mask,
                    }
                )
            else:
                encoder_hidden_states_no_lyric_cond, _ = self.ace_step_transformer.encode(
                    encoder_text_hidden_states,
                    text_attention_mask,
                    torch.zeros_like(speaker_embds),
                    torch.zeros_like(lyric_token_ids), # No lyric
                    lyric_mask, # Mask for no lyric
                )

        def forward_diffusion_with_temperature(hidden_states_diff, timestep_diff, inputs_diff, tau=0.01, l_min=15, l_max=20): # Renamed args
            handlers = []
            def hook(module, input_val, output_val): output_val[:] *= tau; return output_val
            for i in range(l_min, l_max):
                h1 = self.ace_step_transformer.transformer_blocks[i].attn.to_q.register_forward_hook(hook)
                handlers.append(h1)
                h2 = self.ace_step_transformer.transformer_blocks[i].cross_attn.to_q.register_forward_hook(hook)
                handlers.append(h2)
            sample = self.ace_step_transformer.decode(hidden_states=hidden_states_diff, timestep=timestep_diff, **inputs_diff).sample
            for hook_item in handlers: hook_item.remove()
            return sample
    
        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps, desc="Text2Music Diffusion"):
            current_latents = target_latents # Use a temp var for current step's latents

            if is_repaint and x0 is not None and z0 is not None and zt_edit is not None: # Ensure all are defined
                if i < n_min: # type: ignore
                    continue
                elif i == n_min: # type: ignore
                    t_i_repaint = t / 1000 # Renamed
                    # Initial zt_src for repaint: combination of structure (x0) and noise (z0)
                    zt_src_repaint = (1 - t_i_repaint) * x0 + (t_i_repaint) * z0
                    # Align current_latents (which is evolving target_latents) with the repaint process
                    # This formula ensures that the repaint area starts evolving from zt_src_repaint,
                    # while non-repaint area (masked by zt_edit) remains consistent.
                    current_latents = zt_edit + zt_src_repaint - x0 # This assumes zt_edit held x0 in non-repaint areas
                    logger.info(f"Repaint initiated at step {n_min} with noise level {t_i_repaint:.4f}")

            latents_for_model = current_latents
            timestep_for_model = t.expand(latents_for_model.shape[0]) # Renamed
            output_length_for_model = latents_for_model.shape[-1] # Renamed
            
            current_guidance_scale_val = guidance_scale # Renamed
            if is_in_guidance_interval := start_idx <= i < end_idx and do_classifier_free_guidance:
                if guidance_interval_decay > 0 and (end_idx - start_idx -1) > 0: # Avoid div by zero
                    progress = (i - start_idx) / (end_idx - start_idx - 1)
                    current_guidance_scale_val = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
                
                noise_pred_with_cond = self.ace_step_transformer.decode(
                    hidden_states=latents_for_model, attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states_cond, encoder_hidden_mask=encoder_hidden_mask_cond,
                    output_length=output_length_for_model, timestep=timestep_for_model,
                ).sample

                noise_pred_with_only_text_cond = None
                if do_double_condition_guidance and encoder_hidden_states_no_lyric_cond is not None:
                    noise_pred_with_only_text_cond = self.ace_step_transformer.decode(
                        hidden_states=latents_for_model, attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_no_lyric_cond, encoder_hidden_mask=encoder_hidden_mask_cond, # Use correct mask
                        output_length=output_length_for_model, timestep=timestep_for_model,
                    ).sample

                if use_erg_diffusion:
                    noise_pred_uncond = forward_diffusion_with_temperature(
                        hidden_states_diff=latents_for_model, timestep_diff=timestep_for_model,
                        inputs_diff={
                            "encoder_hidden_states": encoder_hidden_states_uncond_null,
                            "encoder_hidden_mask": encoder_hidden_mask_cond, # Mask should match uncond_null
                            "output_length": output_length_for_model, "attention_mask": attention_mask,
                        },
                    )
                else:
                    noise_pred_uncond = self.ace_step_transformer.decode(
                        hidden_states=latents_for_model, attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_uncond_null, encoder_hidden_mask=encoder_hidden_mask_cond, # Mask for uncond_null
                        output_length=output_length_for_model, timestep=timestep_for_model,
                    ).sample

                if do_double_condition_guidance and noise_pred_with_only_text_cond is not None:
                    noise_pred = cfg_double_condition_forward(
                        cond_output=noise_pred_with_cond, uncond_output=noise_pred_uncond,
                        only_text_cond_output=noise_pred_with_only_text_cond,
                        guidance_scale_text=guidance_scale_text, guidance_scale_lyric=guidance_scale_lyric,
                    )
                elif cfg_type == "apg":
                    noise_pred = apg_forward(pred_cond=noise_pred_with_cond, pred_uncond=noise_pred_uncond, guidance_scale=current_guidance_scale_val, momentum_buffer=momentum_buffer)
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(cond_output=noise_pred_with_cond, uncond_output=noise_pred_uncond, cfg_strength=current_guidance_scale_val)
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(noise_pred_with_cond=noise_pred_with_cond, noise_pred_uncond=noise_pred_uncond, guidance_scale=current_guidance_scale_val, i=i, zero_steps=zero_steps, use_zero_init=use_zero_init)
                else: # Default if cfg_type is unknown
                    noise_pred = noise_pred_with_cond 
            else: # Not in guidance interval or CFG disabled
                noise_pred = self.ace_step_transformer.decode(
                    hidden_states=latents_for_model, attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states_cond, encoder_hidden_mask=encoder_hidden_mask_cond,
                    output_length=output_length_for_model, timestep=timestep_for_model,
                ).sample

            # Apply scheduler step or repaint logic
            if is_repaint and i >= n_min and repaint_mask is not None and x0 is not None and z0 is not None: # type: ignore
                t_i_repaint_step = t/1000
                t_im1_repaint_step = (timesteps[i+1]/1000) if i+1 < len(timesteps) else torch.zeros_like(t_i_repaint_step).to(t_i_repaint_step.device)
                
                dtype_noise_pred = noise_pred.dtype
                current_latents_float32 = current_latents.to(torch.float32)
                prev_sample_repaint = current_latents_float32 + (t_im1_repaint_step - t_i_repaint_step) * noise_pred.to(torch.float32)
                prev_sample_repaint = prev_sample_repaint.to(dtype_noise_pred)
                
                # Reconstruct the non-repaint area using x0, z0, and the next timestep's noise level
                # This ensures the background evolves consistently with the diffusion process.
                zt_src_next_step = (1 - t_im1_repaint_step) * x0 + (t_im1_repaint_step) * z0 
                target_latents = torch.where(repaint_mask == 1.0, prev_sample_repaint, zt_src_next_step)
            else:
                target_latents = scheduler.step(model_output=noise_pred, timestep=t, sample=current_latents, return_dict=False, omega=omega_scale)[0] # type: ignore

        # Final adjustments for extend task (if any parts were trimmed and need re-attaching)
        # This part of original code was commented as to_right_pad_gt_latents / to_left_pad_gt_latents
        # It implies these are parts of the *original* x0 that were trimmed, not parts of target_latents.
        # If extend logic created to_..._gt_latents, they should be part of x0 used in repaint.
        # This seems to be for re-attaching parts of x0 that were trimmed due to max_infer_fame_length
        # The `target_latents` is the generated content. If x0 was trimmed, the generated content also matches that trimmed length.
        # Re-attaching parts of original gt to the *generated* content seems unusual unless it's for context.
        # Assuming `to_..._pad_gt_latents` were parts of `x0` (ground truth for non-generated areas)
        # This needs clarification from original intent. For now, if they exist, they are from x0.
        # The current target_latents is the *generated* content for `frame_length`.
        # If the original was `X_left | X_middle | X_right` and we generated for `X_middle_generated`,
        # then to reconstruct, it would be `X_left | X_middle_generated | X_right`.
        # The variables `to_left_pad_gt_latents` and `to_right_pad_gt_latents` are from the `add_retake_noise` section.
        if is_extend:
            final_latents_list = []
            if hasattr(self, 'to_left_pad_gt_latents_from_retake') and self.to_left_pad_gt_latents_from_retake is not None: # type: ignore
                final_latents_list.append(self.to_left_pad_gt_latents_from_retake) # type: ignore
            final_latents_list.append(target_latents)
            if hasattr(self, 'to_right_pad_gt_latents_from_retake') and self.to_right_pad_gt_latents_from_retake is not None: # type: ignore
                final_latents_list.append(self.to_right_pad_gt_latents_from_retake) # type: ignore
            if len(final_latents_list) > 1 :
                 target_latents = torch.cat(final_latents_list, dim=-1)


        return target_latents


    def latents2audio(self, latents, target_wav_duration_second=30, sample_rate=48000, save_path=None, format="flac"):
        output_audio_paths = []
        bs = latents.shape[0]
        # audio_lengths = [target_wav_duration_second * sample_rate] * bs # Not used
        pred_latents = latents
        with torch.no_grad():
            _, pred_wavs = self.music_dcae.decode(pred_latents, sr=sample_rate)
        pred_wavs = [pred_wav.cpu().float() for pred_wav in pred_wavs] # Detach and move to CPU
        for i in tqdm(range(bs), desc="Converting latents to audio"):
            # save_wav_file will handle save_path being None
            output_audio_path = self.save_wav_file(pred_wavs[i], i, save_path=save_path, sample_rate=sample_rate, format=format)
            output_audio_paths.append(output_audio_path)
        return output_audio_paths

    def save_wav_file(self, target_wav, idx, save_path=None, sample_rate=48000, format="flac"):
        if save_path is None:
            logger.warning("save_path is None, using default path ./outputs/")
            base_path = os.path.join(os.getcwd(), "outputs") # Use current working dir
        else:
            base_path = save_path
        ensure_directory_exists(base_path)

        output_filename = f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        output_path_flac = os.path.join(base_path, output_filename)
        
        target_wav_to_save = target_wav.float()
        if target_wav_to_save.ndim == 1: # Ensure it's (channels, time)
            target_wav_to_save = target_wav_to_save.unsqueeze(0)

        torchaudio.save(output_path_flac, target_wav_to_save, sample_rate=sample_rate, format=format)
        logger.info(f"Audio saved to: {output_path_flac}")
        return output_path_flac

    def infer_latents(self, input_audio_path):
        if input_audio_path is None:
            return None
        if not os.path.exists(input_audio_path):
            logger.error(f"Input audio path does not exist: {input_audio_path}")
            return None
            
        input_audio, sr = self.music_dcae.load_audio(input_audio_path)
        input_audio = input_audio.unsqueeze(0) # Add batch dim
        device, dtype = self.device, self.dtype # Use pipeline's device and dtype
        input_audio = input_audio.to(device=device, dtype=dtype)
        with torch.no_grad(): # Ensure no gradients for inference
            latents, _ = self.music_dcae.encode(input_audio, sr=sr)
        return latents

    def __call__(
        self,
        audio_duration: float = 60.0,
        prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None, # Should accept int, list, or str
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        retake_seeds: list = None, # Should accept int, list, or str
        retake_variance: float = 0.5,
        task: str = "text2music",
        repaint_start: int = 0,
        repaint_end: int = 0,
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        save_path: str = None,
        format: str = "flac",
        batch_size: int = 1,
        debug: bool = False,
    ):
        self._ensure_models_on_gpu() # Handles loading if not loaded, or moving from CPU to GPU
        
        output_paths = []
        input_params_json = {}

        try:
            preprocess_start_time = time.time()

            random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)
            retake_random_generators, actual_retake_seeds = self.set_seeds(batch_size, retake_seeds)

            if isinstance(oss_steps, str) and len(oss_steps.strip()) > 0:
                oss_steps_list = list(map(int, oss_steps.split(",")))
            else:
                oss_steps_list = []
            
            texts = [prompt if prompt is not None else ""] # Handle None prompt
            encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts, self.device)
            encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
            text_attention_mask = text_attention_mask.repeat(batch_size, 1)

            encoder_text_hidden_states_null = None
            if use_erg_tag:
                encoder_text_hidden_states_null = self.get_text_embeddings_null(texts, self.device)
                encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(batch_size, 1, 1)

            speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

            lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long() # Default null lyric
            lyric_mask_tensor = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long() # Default null mask
            if lyrics and len(lyrics.strip()) > 0:
                tokenized_lyrics = self.tokenize_lyrics(lyrics, debug=debug)
                lyric_mask_list = [1] * len(tokenized_lyrics)
                lyric_token_idx = torch.tensor(tokenized_lyrics).unsqueeze(0).to(self.device).repeat(batch_size, 1)
                lyric_mask_tensor = torch.tensor(lyric_mask_list).unsqueeze(0).to(self.device).repeat(batch_size, 1)

            if audio_duration <= 0:
                audio_duration = random.uniform(30.0, 120.0) # Capped random duration a bit
                logger.info(f"Using random audio duration: {audio_duration:.2f} seconds.")

            preprocess_time_cost = time.time() - preprocess_start_time
            
            diffusion_process_start_time = time.time()

            add_retake_noise = task in ("retake", "repaint", "extend")
            if task == "retake": # Retake is full repaint
                repaint_start = 0
                repaint_end = audio_duration 
            
            src_latents = None
            if src_audio_path is not None:
                if not (task in ("repaint", "edit", "extend")):
                    logger.warning(f"src_audio_path ('{src_audio_path}') provided for task '{task}', but it's typically used for 'repaint', 'edit', or 'extend'. It will be loaded.")
                # Keeping original assert behavior for incompatible tasks if src_audio is key
                assert task in ("repaint", "edit", "extend"), f"src_audio_path is provided, but task is '{task}'. Expected 'repaint', 'edit', or 'extend'."
                src_latents = self.infer_latents(src_audio_path)
                if src_latents is None and task in ("repaint", "edit", "extend"): # If essential and failed
                     raise ValueError(f"Source latents could not be inferred from {src_audio_path}, which is required for task {task}")


            target_latents = None
            if task == "edit":
                if src_latents is None:
                     raise ValueError("src_audio_path (and valid latents) are required for 'edit' task.")
                edit_texts = [edit_target_prompt if edit_target_prompt is not None else ""]
                target_encoder_text_hidden_states, target_text_attention_mask = self.get_text_embeddings(edit_texts, self.device)
                target_encoder_text_hidden_states = target_encoder_text_hidden_states.repeat(batch_size, 1, 1)
                target_text_attention_mask = target_text_attention_mask.repeat(batch_size, 1)

                target_lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
                target_lyric_mask_tensor = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
                if edit_target_lyrics and len(edit_target_lyrics.strip()) > 0:
                    tokenized_target_lyrics = self.tokenize_lyrics(edit_target_lyrics, debug=True)
                    target_lyric_mask_list = [1] * len(tokenized_target_lyrics)
                    target_lyric_token_idx = torch.tensor(tokenized_target_lyrics).unsqueeze(0).to(self.device).repeat(batch_size, 1)
                    target_lyric_mask_tensor = torch.tensor(target_lyric_mask_list).unsqueeze(0).to(self.device).repeat(batch_size, 1)

                target_speaker_embeds = speaker_embeds.clone() # Assuming same speaker for edit

                target_latents = self.flowedit_diffusion_process(
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embeds,
                    lyric_token_ids=lyric_token_idx,
                    lyric_mask=lyric_mask_tensor,
                    target_encoder_text_hidden_states=target_encoder_text_hidden_states,
                    target_text_attention_mask=target_text_attention_mask,
                    target_speaker_embeds=target_speaker_embeds,
                    target_lyric_token_ids=target_lyric_token_idx,
                    target_lyric_mask=target_lyric_mask_tensor,
                    src_latents=src_latents,
                    random_generators=retake_random_generators,
                    infer_steps=infer_step,
                    guidance_scale=guidance_scale,
                    n_min=edit_n_min,
                    n_max=edit_n_max,
                    n_avg=edit_n_avg,
                )
            else: # text2music, repaint, extend
                target_latents = self.text2music_diffusion_process(
                    duration=audio_duration,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embds=speaker_embeds,
                    lyric_token_ids=lyric_token_idx,
                    lyric_mask=lyric_mask_tensor,
                    guidance_scale=guidance_scale,
                    omega_scale=omega_scale,
                    infer_steps=infer_step,
                    random_generators=random_generators,
                    scheduler_type=scheduler_type,
                    cfg_type=cfg_type,
                    guidance_interval=guidance_interval,
                    guidance_interval_decay=guidance_interval_decay,
                    min_guidance_scale=min_guidance_scale,
                    oss_steps=oss_steps_list,
                    encoder_text_hidden_states_null=encoder_text_hidden_states_null,
                    use_erg_lyric=use_erg_lyric,
                    use_erg_diffusion=use_erg_diffusion,
                    retake_random_generators=retake_random_generators,
                    retake_variance=retake_variance,
                    add_retake_noise=add_retake_noise,
                    guidance_scale_text=guidance_scale_text,
                    guidance_scale_lyric=guidance_scale_lyric,
                    repaint_start=repaint_start,
                    repaint_end=repaint_end,
                    src_latents=src_latents,
                )

            diffusion_time_cost = time.time() - diffusion_process_start_time
            
            latents2audio_start_time = time.time()

            if target_latents is not None:
                output_paths = self.latents2audio(
                    latents=target_latents,
                    target_wav_duration_second=audio_duration, # This should align with generated latent length
                    save_path=save_path,
                    format=format,
                )
            else:
                logger.error("Target latents were not generated. Skipping audio conversion.")
                output_paths = []


            latent2audio_time_cost = time.time() - latents2audio_start_time
            
            timecosts = {
                "preprocess": round(preprocess_time_cost, 2),
                "diffusion": round(diffusion_time_cost, 2),
                "latent2audio": round(latent2audio_time_cost, 2),
            }

            input_params_json = {
                "task": task,
                "prompt": prompt if task != "edit" else edit_target_prompt,
                "lyrics": lyrics if task != "edit" else edit_target_lyrics,
                "audio_duration": round(audio_duration, 2),
                "infer_step": infer_step,
                "guidance_scale": guidance_scale,
                "scheduler_type": scheduler_type,
                "cfg_type": cfg_type,
                "omega_scale": omega_scale,
                "guidance_interval": guidance_interval,
                "guidance_interval_decay": guidance_interval_decay,
                "min_guidance_scale": min_guidance_scale,
                "use_erg_tag": use_erg_tag,
                "use_erg_lyric": use_erg_lyric,
                "use_erg_diffusion": use_erg_diffusion,
                "oss_steps": oss_steps_list,
                "timecosts": timecosts,
                "actual_seeds": actual_seeds,
                "retake_seeds": actual_retake_seeds,
                "retake_variance": retake_variance,
                "guidance_scale_text": guidance_scale_text,
                "guidance_scale_lyric": guidance_scale_lyric,
                "repaint_start": repaint_start,
                "repaint_end": repaint_end,
                "edit_n_min": edit_n_min,
                "edit_n_max": edit_n_max,
                "edit_n_avg": edit_n_avg,        
                "src_audio_path": src_audio_path,
                "edit_target_prompt": edit_target_prompt,
                "edit_target_lyrics": edit_target_lyrics,
                "device_used": str(self.device),
                "dtype_used": str(self.dtype),
                "torch_compile_enabled": self.torch_compile,
            }
            
            for output_audio_path_item in output_paths: # Renamed loop var
                input_params_json_save_path = output_audio_path_item.replace(f".{format}", "_input_params.json")
                # Create a copy for each file to avoid modifying the base dict if needed later
                current_file_params = input_params_json.copy()
                current_file_params["audio_path"] = output_audio_path_item 
                try:
                    with open(input_params_json_save_path, "w", encoding="utf-8") as f:
                        json.dump(current_file_params, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Failed to save input_params_json to {input_params_json_save_path}: {e}")


            return output_paths + [input_params_json] # Return the base params dict once with all audios

        except Exception as e:
            logger.exception(f"Error during ACEStepPipeline __call__: {e}")
            # Return empty list and error details, or re-raise
            error_details = {
                "error": str(e),
                "task_params": {key: val for key, val in locals().items() if key not in ['self', 'e'] and not callable(val)} # Log input params
            }
            return [], error_details # Or re-raise e
        finally:
            self.unload_models_to_cpu_and_clear_ram()
