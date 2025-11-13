# please refer to ddim_sampling()
import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)

        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        self.register_buffer('ddim_sqrt_one_minus_alphas_prev', np.sqrt(1. - ddim_alphas_prev))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform', #uniform_trailing for starting from last timestep
               guidance_rescale=0.0,
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=schedule_verbose)

        if len(mask) > 1 and mask[1].shape[1] > 16:
            batch_size = 2
        elif (mask[0][1] - mask[0][0] + 1) > 16:
            batch_size = 2

        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        device = self.model.betas.device 
        
        
        

        samples = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    precision=precision,
                                                    fs=fs,
                                                    guidance_rescale=guidance_rescale,model_flow=None,#model_without_ddp,
                                                    **kwargs)

        return samples

    def cos_loss(self,latent0,guidance):
        error = torch.abs(latent0-guidance).mean()
        return error
    def cosine_similarity(self, a, b):
        sum_dot = 0
        pow_sum_k = 0
        pow_sum_ki = 0
        v_k = a[0]
        v_k_i = b[0]
        print(v_k.shape)
        n_channel, time,_,_ = v_k.shape  # 通道数，维度
        d_total = 0
        for t in range(time):
            for m in range(n_channel):
                sum_dot += torch.dot(v_k[m][t].flatten(), v_k_i[m][t].flatten().T)
                pow_sum_k += torch.dot(v_k[m][t].flatten(), v_k[m][t].flatten().T)
                pow_sum_ki += torch.dot(v_k_i[m][t].flatten(), v_k_i[m][t].flatten().T)
            d = (sum_dot / torch.sqrt(pow_sum_k * pow_sum_ki)).float()
            d_total += d
        return d_total
    
    @torch.no_grad()
    def q_inverse_ddim(self, x, c, t, index, use_original_steps=False, quantize_denoised=False,
                      score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      guidance_rescale=0.0,**kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            ### do_classifier_free_guidance
            if isinstance(c, torch.Tensor) or isinstance(c, dict):
                e_t_cond = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_alphas_prev = self.ddim_sqrt_one_minus_alphas_prev
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)
        sqrt_one_minus_aprev = torch.full(size, sqrt_one_minus_alphas_prev[index],device=device)
        # current prediction for x_0

        pred_x0 = (x - sqrt_one_minus_aprev * e_t) / a_prev.sqrt()

        
        if self.model.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
            rescale = (prev_scale_t / scale_t)
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        
    
        x_prev = a_t.sqrt() * pred_x0 + sqrt_one_minus_at * e_t

        return x_prev, pred_x0
    @torch.no_grad()
    def ddim_inversion(self, x0, cond, timesteps,total_steps,device,b,use_original_steps,quantize_denoised,
                       score_corrector,corrector_kwargs,unconditional_guidance_scale,unconditional_conditioning,
                       guidance_rescale):
        time_range = timesteps#np.flip(timesteps)
        for i, t in enumerate(tqdm(time_range, desc="DDIM inversion",total=total_steps)):
            index = i #total_steps - i - 1
            ts = torch.full((b,), t, device=device, dtype=torch.long)

            outs = self.q_inverse_ddim(x0, cond, ts, index, use_original_steps, quantize_denoised,
                      score_corrector, corrector_kwargs,
                      unconditional_guidance_scale, unconditional_conditioning,
                      guidance_rescale)
            x0, pred_x0 = outs

        return x0

    @torch.no_grad()
    def slerp(self, latents0, latents1, fract_mixing):
        r""" Copied from lunarring/latentblending
        Helper function to correctly mix two random variables using spherical interpolation.
        The function will always cast up to float64 for sake of extra 4.
        Args:
            p0: 
                First tensor for interpolation
            p1: 
                Second tensor for interpolation
            fract_mixing: float 
                Mixing coefficient of interval [0, 1]. 
                0 will return in p0
                1 will return in p1
                0.x will return a mix between both preserving angular velocity.
        """
        if latents0 is None or latents1 is None:
            return latents0 if latents1 is None else latents1
        p0 = latents0
        p1 = latents1
        if p0.dtype == torch.float16:
            recast_to = 'fp16'
        else:
            recast_to = 'fp32'

        p0 = p0.double()
        p1 = p1.double()
            
        norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
        epsilon = 1e-7
        dot = torch.sum(p0 * p1) / norm
        dot = dot.clamp(-1+epsilon, 1-epsilon)

        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * fract_mixing
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = torch.sin(theta_t) / sin_theta_0
        interp = p0*s0 + p1*s1

        if recast_to == 'fp16':
            interp = interp.half()
        elif recast_to == 'fp32':
            interp = interp.float()

        return interp

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,precision=None,fs=None,guidance_rescale=0.0,model_flow=None,
                      **kwargs):
        device = self.model.betas.device        
        b = shape[0]
        if x_T is None:
            img_all = torch.randn(shape, device=device)
        # DDIM inverse
        else:
            img_all = x_T
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img_all[:1]], 'pred_x0': [img_all[:1]]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]


        clean_cond = kwargs.pop("clean_cond", False)

        iterator_list = []

        iterator = time_range
        iterator2 = time_range
        iterator_list = [iterator,iterator2]

        alphas = self.model.alphas_cumprod 
        alphas_prev = self.model.alphas_cumprod_prev 
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod 
        sigmas = self.ddim_sigmas_for_original_num_steps 
        size = (b, 1, 1, 1, 1)

        # dual-step generation
        traj_all = mask
        t_traj = 16
        if len(mask) > 1:
            t_traj = mask[1].shape[1]
        else:
            t_traj = mask[0][1] - mask[0][0] + 1
        delta_t_traj = t_traj - 16
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        img_first = torch.zeros((total_steps,1,shape[1],1,shape[3],shape[4]),device=device)
        cond_text = 0
        for batch_index in range(b):
            if batch_index==1:
                img_cond = cond["c_crossattn"][0][:,77:,:,:]
                cond_text = cond["c_crossattn"][0][:,0:77,:,:]
                text_1 = cond_text[:,:,delta_t_traj]
                text_2 = cond_text[:,:,15]
                alpha_list = list(torch.linspace(0, 1, 16))
                time_index = 0
                for alpha in alpha_list:
                    cond_text[:,:,time_index] = self.slerp(text_1, text_2, alpha)
                    time_index += 1
                cond["c_crossattn"][0][:,0:77,:,:] = cond_text
            if batch_index==1:

                cond['c_concat'][0][:,:,0,:,:] = img_all[0,:,delta_t_traj,:,:]


            loss_min = -1
            img_best = 0

            for delta_cond in [30]:
                sigm = 0
                print(f"delta cod {delta_cond}")
                img = img_all[batch_index:batch_index+1]
                pred_x0 = img_all[batch_index:batch_index+1]
                loss_taj = 0

                for i, step in enumerate(iterator_list[batch_index]):
                    index = total_steps - i - 1
                    ts = torch.full((img.shape[0],), step, device=device, dtype=torch.long)

                    a_t = torch.full(size, alphas[index], device=device)[:1]
                    a_prev = torch.full(size, alphas_prev[index], device=device)[:1]
                    sigma_t = torch.full(size, sigmas[index], device=device)[:1]
                    sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)[:1]

                    cond_grad = torch.zeros_like(img)
                    with torch.enable_grad():
                        width = img.shape[4]
                        height = img.shape[3]

                        t = img.shape[2]
             
                        for batch_id in range(1):
                            if len(mask) == 1:
                                break
                            latent_features = pred_x0.requires_grad_()

                            los = 0


                            def rescale(x,y):
                                h = 320
                                w = 512
                                new_x = int(x)#int(x / w * width) if int(x / w * width) < width else width - 1 # int(x) if x < width else width - 1# 
                                new_y = int(y)#int(y / h * height) if int(y / h * height) < height else height - 1 # int(y) if y < height else height-1#
                                # print(new_x,new_y)
                                return [new_x,new_y]

                            for trj_info_idx in traj_all[1:]:
                                trj_idx = trj_info_idx
       
                                trj_cur = np.zeros((trj_idx.shape[0],16,2))
                                trj_cur[:,0] = trj_idx[:,0]
                                trj_cur[:,15] = trj_idx[:,t_traj-1]

                                if delta_t_traj > 0:
                                    if batch_index == 1:
                                        trj_cur[:,1:15] = trj_idx[:,delta_t_traj+1:t_traj-1]
              

                                    if batch_index == 0:
                                        trj_cur[:,1:15] = trj_idx[:,1:15]

                                else:
                                    trj_cur = trj_idx
                                los = 0
                                b_los = 0
                                for dot in trj_cur:
                                    for idx in range(0,t-2):
                                        total = t - 2
                                        factor1 = 0
                                        factor2 = 0
                                        mask_step_w = 2
                                        mask_step_h = 2 
                                        if 0 < dot[0][0]// 8  < width and 0 < dot[0][1]// 8  < height:
                                            if dot[0][0] // 8 - mask_step_w < 0 or dot[0][0] // 8 + mask_step_w >= width:
                                                mask_step_w = 0
                                            if dot[0][1] // 8 - mask_step_h < 0 or dot[0][1] // 8 + mask_step_h >= height:
                                                mask_step_h = 0                                         
                                            factor1 = 1 if idx + 1 < (t-2) / 2 else 0
                                            factor1 = 0.5 if (t-2) / 2 -1 <= idx + 1 <= (t-2) / 2 + 1 else factor1
                                        if 0 < dot[t-1][0] // 8 < width and 0 < dot[t-1][1] // 8 < height:

                                            if dot[t-1][0] // 8 - mask_step_w < 0 or dot[t-1][0] // 8 + mask_step_w >= width:
                                                mask_step_w = 0
                                            if dot[t-1][1] // 8 - mask_step_h < 0 or dot[t-1][1] // 8 + mask_step_h >= height:
                                                mask_step_h = 0   
                                            factor2 = 0 if idx + 1 < (t-2) / 2 else 1
                                            factor2 = 0.5 if (t-2) / 2 -1 <= idx + 1 <= (t-2) / 2 + 1 else factor2
                                        factor1 = 1 if ((idx + 1 > (t-2) / 2) and (factor2 == 0) and (factor1 != 0)) else factor1 
                                        factor2 = 1 if ((idx + 1 <= (t-2) / 2) and (factor1 == 0) and (factor2 != 0)) else factor2

                                        # 密集轨迹引导  

                                        if 0 < dot[idx+1][0]//8 < width and 0 < dot[idx+1][1]//8 < height:
                                            if dot[idx+1][0] // 8 - mask_step_w < 0 or dot[idx+1][0] // 8 + mask_step_w >= width:
                                                mask_step_w = 0
                                            if dot[idx+1][1] // 8 - mask_step_h < 0 or dot[idx+1][1] // 8 + mask_step_h >= height:
                                                mask_step_h = 0 
                                                
                                            traj = rescale(dot[idx+1][0] // 8,dot[idx+1][1]// 8) 
                                            new_y = traj[1]
                                            new_x = traj[0]

                                            f_next = latent_features[:1,:,idx+1,new_y-mask_step_h:new_y+mask_step_h+1,new_x-mask_step_w:new_x+mask_step_w+1]

                                            f_first = f_next
                                            f_last = f_next

                                            if 0 < dot[0][0] //8 < width and 0 < dot[0][1]//8 < height and factor1 != 0:
                                                traj = rescale(dot[0][0] // 8,dot[0][1]// 8) 
                                                if batch_index==0:
                                                    f_first = latent_features[:1,:,0,traj[1]-mask_step_h:traj[1]+mask_step_h+1,traj[0]-mask_step_w:traj[0]+mask_step_w+1]
                                                else:
                                                    f_first = img_first[i,:1,:,0,traj[1]-mask_step_h:traj[1]+mask_step_h+1,traj[0]-mask_step_w:traj[0]+mask_step_w+1]

                                            if 0 < dot[t-1][0]//8 < width and 0 < dot[t-1][1]//8 < height and factor2 != 0:
                                                traj = rescale(dot[t-1][0] // 8,dot[t-1][1]// 8) 
            
                                                f_last = latent_features[:1,:,15,traj[1]-mask_step_h:traj[1]+mask_step_h+1,traj[0]-mask_step_w:traj[0]+mask_step_w+1]
         
                                            cos_los = torch.sum(  factor1* torch.abs(f_first.detach() - f_next) +  factor2  * torch.abs(f_last.detach() - f_next))
                                           
                                            los += cos_los
           
                                            
                                            if i == total_steps - 1:
                                                loss_taj += cos_los
        
                                if los > 0:
                                    cond_grad += -torch.autograd.grad(los, latent_features)[0][0]

                            cond_mean = cond_grad.max()

                                
                            if i == 0 and cond_mean != 0:
                                sigm = delta_cond
                            # classifier guidance
                            cond_grad= ((cond_grad) *  sigm * (1 - a_t))


                        if batch_index==0:
                            img_first[i,0,:,0] = img[0,:,0]
                        img = img + cond_grad 
                        outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    mask=cond_grad,x0=x0,fs=fs,guidance_rescale=guidance_rescale,
                                                    **kwargs)
                            
                        img, pred_x0 ,e_t, noise = outs

                if len(mask) == 1:
                    img_best = img
                    break
                
                if loss_min < 0 or loss_min > loss_taj:
                    if loss_min < 0:
                        loss_min = loss_taj
                        img_best = img
                    else:
                        loss_min = loss_taj
                        img_best = img


            img_all[batch_index:batch_index+1] = img_best
            

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img_all

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None,mask=None,x0=None,guidance_rescale=0.0,**kwargs):
        b, *_, device = *x.shape, x.device

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas

        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:

            if isinstance(c, torch.Tensor) or isinstance(c, dict):
 
                e_t_cond = self.model.apply_model(x, t, c, **kwargs)

                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning,uncondition=1, **kwargs)
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)


            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)
  
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)

        else:
            e_t = model_output

        

        if score_corrector is not None:

            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)



        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
        
        if self.model.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
            rescale = (prev_scale_t / scale_t)
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return x_prev, pred_x0, e_t, noise#, feature_map

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
