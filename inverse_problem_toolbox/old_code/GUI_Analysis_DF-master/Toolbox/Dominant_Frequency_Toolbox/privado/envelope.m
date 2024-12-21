function [envUpper,envLower] = envelope(indMax,indMin,x,t);


if(nargin < 4 || isempty(t))
    t = 1:length(x);
end

%Tomamos como primer y ultimo índice los puntos extremos de la señal.
%Nota: Cambiar por algo más combeniente
%indMaxAux = [1,indMax,length(x)];
%indMinAux = [1,indMin,length(x)];

%Cambiamos las condiciones en los extremos por lo que se sugiere en el
%documento de Rilling et al.
[tmin,tmax,xmin,xmax] = boundary_conditions(indMin,indMax,t,x);
    

%Calculamos los puntos de las envolventes
%envUpper = interp1q(t(indMaxAux),x(indMaxAux),t,'spline');
%envLower = interp1q(t(indMinAux),x(indMinAux),t,'spline');

%Con los cambios propuestos por las condiciones de extremos se debe operar
%asi

 envUpper = interp1(tmax,xmax,t,'spline');
 envLower = interp1(tmin,xmin,t,'spline');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%Funciones auxiliares%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%Boundary Conditions

% G. Rilling, P. Flandrin and P. Gonçalvès
% "On Empirical Mode Decomposition and its algorithms",
% IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
% NSIP-03, Grado (I), June 2003


function [tmin,tmax,xmin,xmax] = boundary_conditions(indmin,indmax,t,x,nbsym)
% computes the boundary conditions for interpolation (mainly mirror symmetry)
if(size(x,2) == 1)
    x = x';
end
%Por defecto se utiliza nbsym = 2

nbsym = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
	
	lx = length(x);
	
	if (length(indmin) + length(indmax) < 3)
		error('not enough extrema')
	end

	if indmax(1) < indmin(1)
    	if x(1) > x(indmin(1))
			lmax = fliplr(indmax(2:min(end,nbsym+1)));
			lmin = fliplr(indmin(1:min(end,nbsym)));
			lsym = indmax(1);
		else
			lmax = fliplr(indmax(1:min(end,nbsym)));
			lmin = [fliplr(indmin(1:min(end,nbsym-1))),1];
			lsym = 1;
		end
	else

		if x(1) < x(indmax(1))
			lmax = fliplr(indmax(1:min(end,nbsym)));
			lmin = fliplr(indmin(2:min(end,nbsym+1)));
			lsym = indmin(1);
		else
			lmax = [fliplr(indmax(1:min(end,nbsym-1))),1];
			lmin = fliplr(indmin(1:min(end,nbsym)));
			lsym = 1;
		end
	end
    
	if indmax(end) < indmin(end)
		if x(end) < x(indmax(end))
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
			rmin = fliplr(indmin(max(end-nbsym,1):end-1));
			rsym = indmin(end);
		else
			rmax = [lx,fliplr(indmax(max(end-nbsym+2,1):end))];
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
			rsym = lx;
		end
	else
		if x(end) > x(indmin(end))
			rmax = fliplr(indmax(max(end-nbsym,1):end-1));
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
			rsym = indmax(end);
		else
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
			rmin = [lx,fliplr(indmin(max(end-nbsym+2,1):end))];
			rsym = lx;
		end
	end
    
	tlmin = 2*t(lsym)-t(lmin);
	tlmax = 2*t(lsym)-t(lmax);
	trmin = 2*t(rsym)-t(rmin);
	trmax = 2*t(rsym)-t(rmax);
    
	% in case symmetrized parts do not extend enough
	if tlmin(1) > t(1) | tlmax(1) > t(1)
		if lsym == indmax(1)
			lmax = fliplr(indmax(1:min(end,nbsym)));
		else
			lmin = fliplr(indmin(1:min(end,nbsym)));
		end
		if lsym == 1
			error('bug')
		end
		lsym = 1;
		tlmin = 2*t(lsym)-t(lmin);
		tlmax = 2*t(lsym)-t(lmax);
	end   
    
	if trmin(end) < t(lx) | trmax(end) < t(lx)
		if rsym == indmax(end)
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
		else
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
		end
	if rsym == lx
		error('bug')
	end
		rsym = lx;
		trmin = 2*t(rsym)-t(rmin);
		trmax = 2*t(rsym)-t(rmax);
	end 
          
	xlmax =x(lmax); 
	xlmin =x(lmin);
	xrmax =x(rmax); 
	xrmin =x(rmin);
     
	tmin = [tlmin t(indmin) trmin];
	tmax = [tlmax t(indmax) trmax];
	xmin = [xlmin x(indmin) xrmin];
	xmax = [xlmax x(indmax) xrmax];