
;Computes 3D dct using multiple 1D transforms
;Wrapper for dct_fft and should effectively calculate DCT for any number of dimensions
;
;
;
;HISTORY: Created RJ Morton 09/2018
;
FUNCTION dct_fft_3d, cube,_extra=extra

sz=size(cube)
siz=sz

temp=cube
FOR i=0,sz(0)-1 DO BEGIN
	out=dct_fft(temp,_extra=extra,/quiet) ; computes 1D transform
	temp=transpose(out,[[1:sz(0)-1],0]) ; circular shift of array
ENDFOR

out=temp
return,out
END