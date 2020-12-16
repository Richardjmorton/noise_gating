;Performs noise-gating of images
;
;
;INPUT - data cube
;
;OUTPUT - noise-gated data cube
;
;OPTIONAL INPUTS - gate - use gating (default is Weiner Filter)
;                  dct - use Discrete Fourier Transform (much slower)
;                  perc - percentage level of samples to calculate noise level, default 0.5 (median)
;                  gamma - level of SNR to use as a cut-off value, default is 1.5
;                  image_indepen_noise - calculation for removal of image independent additive noise
;
;
;TO DO - Parallel-ise 
;
;HISTORY:
;      Written by RJ Morton 09/18 following description in DeForest ApJ, 838, 155, 2017
;
;      Craig DeForest asked me to add this note:
;      Underlying algorithm: patent applied for in 2016 by Southwest Research Institute
;      Use for space research is unrestricted


;estimates shot noise
FUNCTION estimate_shot_noise, fourier_image, image 
    fac=total(sqrt(image>0))
    betas=abs(fourier_image)/fac
    return,betas
END

;estimates image independent noise
FUNCTION image_indepen_noise, fourier_image
    betas=abs(fourier_image)
    return,betas
END

;Implementation of gating
FUNCTION gate, fourier_image, betas, gamm=gamm
    IF NOT keyword_set(gamm) THEN gamm=1.5
    filt = (1./gamm)*abs(fourier_image)/betas
    filt = temporary(filt) gt 1
    return,filt
END

;Implementation of Wiener filter
FUNCTION wiener_filt, fourier_image, betas, gamm=gamm
    IF NOT keyword_set(gamm) THEN gamm=1.5
    filt=abs(fourier_image)/(gamm*betas+abs(fourier_image))
    return, filt
END



;Main code
PRO noise_gate, data, gated_data, gate=gate,dct=dct,beta_var=beta_var,$ 
                image_indepen_noise=image_indepen_noise, _extra=_extra


sz=size(data)
nx=sz(1) & ny=sz(2) & nz=sz(3)

win_size = 12 ;window size of sub-images
half_win = win_size/2

;apodisation cube for sub-images
apod=do_apod3d(win_size,win_size,win_size,/squared) 


noise_elem = nx*ny / win_size / win_size
noise_arr = fltarr(win_size,win_size,win_size,noise_elem) ;scratch array

;Loop for calculating noise profile
h=0
print,"Calculating noise model..."
FOR i=half_win, nx-half_win-1, win_size DO $
FOR j=half_win, ny-half_win-1, win_size DO BEGIN

    x = [i-half_win, i+half_win-1]
    y = [j-half_win, j+half_win-1] 
	subImage = data[x[0]:x[1],y[0]:y[1],0:win_size-1]*apod ;sub-image & apod
	
    IF keyword_set(dct) THEN fourier_image = dct_fft_3d(subImage,/ortho) $ ;Discrete Cosine transform - orthonormal
                        ELSE fourier_image = fft(subImage) 
    
	
	IF NOT keyword_set(image_indepen_noise) THEN temp_betas = estimate_shot_noise(fourier_image, subImage) $ 
	ELSE temp_betas = image_indepen_noise(fourier_image)

    noise_arr[0:-1,0:-1,0:-1,h]=temp_betas
    h=h+1

ENDFOR

;Set noise profile
IF NOT keyword_set(perc) THEN perc=0.5 ;default is median

IF keyword_set(beta_var) THEN BEGIN
    betas=fltarr(win_size,win_size,win_size)

    FOR i=0,win_size-1 DO $ 
    FOR j=0,win_size-1 DO $ 
    FOR k=0,win_size-1 DO BEGIN

      strip = noise_arr[i,j,k,0:h-1]
      betas[i,j,k] = (strip(sort(abs(strip))))[perc*noise_elem]

    ENDFOR

ENDIF ELSE BEGIN
    ;One value of beta - seems to work best!
    betas=(noise_arr[sort(noise_arr[*,*,*,0:h-1])])[perc*n_elements(noise_arr[*,*,*,0:h-1])]
ENDELSE


os_skip=win_size/4 ; using sin^4 window
gated_data=fltarr(nx,ny,nz) ;scratch array

print,'Implementing noise gating...'

FOR i=half_win, nx-half_win-1, os_skip DO $
FOR j=half_win, ny-half_win-1, os_skip DO $
FOR k=half_win, nz-half_win-1, os_skip DO BEGIN $

    x = [i-half_win, i+half_win-1]
    y = [j-half_win, j+half_win-1] 
    z = [k-half_win, k+half_win-1]
    im = data[x[0]:x[1], y[0]:y[1], z[0]:z[1]]*apod
    
    IF keyword_set(dct) THEN fourier_image=dct_fft_3d(im,/ortho) $
                           ELSE fourier_image=fft(im)

    IF NOT keyword_set(image_indepen_noise) THEN noise_profile = betas*total(sqrt(im>0)) $
                                            ELSE noise_profile = betas
    
    IF NOT keyword_set(gate) THEN filt = wiener_filt(fourier_image, noise_profile , _extra=_extra) $ 
                             ELSE filt = gate(fourier_image, noise_profile , _extra=_extra)
    
    ; Keeping core region of DCT when using varying beta
    IF keyword_set(beta_var) THEN BEGIN
        filt = shift(filt, half_win, half_win, half_win)
        filt[5:7,5:7,5:7] = 1
        filt = shift(filt, half_win, half_win, half_win)
    ENDIF
    
    IF keyword_set(dct) THEN BEGIN 
        inverse_dct = dct_fft_3d(fourier_image*filt, /inverse, /ortho)*apod
        gated_data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] += inverse_dct

    ENDIF ELSE BEGIN
        inverse_ft = real_part(fft(fourier_image*filt,/inverse)*apod)
        gated_data[x[0]:x[1], y[0]:y[1], z[0]:z[1]] += inverse_ft
    ENDELSE
ENDFOR

;correction factor for windowing
;only applicable for sin^4 window
gated_data/=(1.5)^3

END