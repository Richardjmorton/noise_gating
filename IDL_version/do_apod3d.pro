
; define 3D apodisation 
;
; As given in DeForest 2017
;
;R Morton 2014

FUNCTION do_apod3d,nx,ny,nz,squared=squared
 

IF keyword_set(squared) THEN BEGIN
	apodx = (sin(!pi*(findgen(nx)+0.5)/nx))^2
	apody = (sin(!pi*(findgen(ny)+0.5)/ny))^2
	apodz = (sin(!pi*(findgen(nz)+0.5)/nz))^2

	apodxy = apodx # apody
	apodxyz=rebin(apodxy,nx,ny,nz)
	apodz2=transpose(rebin(apodz,nz,nx,ny),[1,2,0])

	apodxyz*=apodz2
ENDIF ELSE BEGIN

    ;some issue with this part! 
    apodx = (sin(!pi*(findgen(nx)+0.5)/nx))
	apody = (sin(!pi*(findgen(ny)+0.5)/ny))
	apodz = (sin(!pi*(findgen(nz)+0.5)/nz))

	apodxy = apodx # apody
	apodxyz=rebin(apodxy,nx,ny,nz)
	apodz2=transpose(rebin(apodz,nz,nx,ny),[1,2,0])
    apodxyz*=apodz2

ENDELSE



RETURN,apodxyz
END