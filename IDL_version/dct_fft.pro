;Purpose - Calculate fast 1D Discrete Cosine Transform using FFT
;
; Uses formula C(k)=2*sum_{n=0}^{N-1} x(n) cos ({\pi k}{2N}(2n+1))
; For comparison to MATLAB - they use C(k)=w(k)*sum_{n=0}^{N-1} x(n) cos ({\pi k}{2N}(2n+1))
;                            where w(0)=\sqrt(1./N) and w(k \neq 0)=\sqrt(2/N)
;
;Returns double precision array of forward/inverse transformed series
;
;Note: accurate to ~10^-14, i.e., original series minus a forward then inverse'd series
;      changing from double to float precision leads to ~10^-6 accuracy
;
;HISTORY: Created RJ Morton 09/2018
;

FUNCTION dct_fft, series,inverse=inverse,ortho=ortho,quiet=quiet


sz=size(series)
N=sz(1)
k=indgen(N)
 
 grid_size=[sz[1:sz(0)]] ;if input is greater than 1D
 shift_grid=exp( rebin(k,grid_size)*dcomplex(0,-1.*!dpi/(2.*N) ) ) 
 ind=[k,reverse(k)]
 ind=ind(2*k)

IF NOT keyword_set(quiet) THEN IF sz(0) GT 1 THEN print, 'Only DCT-ing first dimension'

IF NOT keyword_set(inverse) THEN BEGIN

   CASE sz(0) OF 
      1:even_ser=series(ind)
      2:even_ser=series(ind,*)
      3:even_ser=series(ind,*,*)
   ENDCASE
   ft_ser=(fft(even_ser,dim=1,/double))
   dct=2.*N*real_part(shift_grid*ft_ser)
    
    if keyword_set(ortho) then begin
		dct[0] *= sqrt(1/2.)
		dct /= sqrt(2*N)
	endif


ENDIF ELSE BEGIN

	temp_series=series
	if keyword_set(ortho) then begin
		temp_series[0] /= sqrt(1/2.)
		temp_series *= sqrt(2.*N)
	endif

	CASE sz(0) OF 
     1: ck=dcomplex(temp_series,-temp_series(N-k))
     2: ck=dcomplex(temp_series,-temp_series(N-k,0:-1))
     3: ck=dcomplex(temp_series,-temp_series(N-k,0:-1,0:-1))
    ENDCASE 

  	rev_ser=0.5/shift_grid/N*ck
	ser=fft(rev_ser,/inverse,dim=1,/double)
    ind=[k,reverse(k)]
    ind=ind(2*k)

    CASE sz(0) OF 
     1: dct=dblarr(N)
     2: dct=dblarr(N,sz(1))
     3: dct=dblarr(N,sz(1),sz(2))
    ENDCASE 

    CASE sz(0) OF 
     1:  dct[ind]=ser
     2:  dct[ind,0:-1]=ser
     3:  dct[ind,0:-1,0:-1]=ser
    ENDCASE 
   

ENDELSE

return,dct

END