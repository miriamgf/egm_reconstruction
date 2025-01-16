function [EJ,row,col] = bemEJMatrix(model,surf1,surf2,mode)
% Algoritmo que procesa la influencia de la divergencia del potencial 
% de un node a otro node.

% La aproximacion de la integracion se hace mediante cuadratura de Gauss:
% Orden 5, 7 puntos en el triangulo a integrar.
%
% J. Radon, "Zur mechanischen Kubatur", Monatshefte für Mathematik, 1948,
% 52:286-300.
% http://www.digizeitschriften.de/de/dms/img/#navi
%
% D.A. Dunavant, "High Degree Efficient Symmetrical Gaussian Quadrature 
% Rules For the Triangle", 1985, International Journal for Numerical 
% Methods in Engineering, 21:1129-1148.
%
% G.R. Cowper, "Gaussian Quadrature Formulas For Triangles", 1972,
% International Journal for Numerical Methods in Engineering, 7(3):405-408.

    rowstart = 0;
    row =[];
    for p = 1:length(surf1),
        rows{p} = rowstart + [1:size(model.surface{surf1(p)}.node,2)];
        rowstart = rows{p}(end);
        row = [ row surf1(p)*ones(size(rows{p}))];
    end    

    colstart = 0;
    col = [];
    for p = 1:length(surf2),
        cols{p} = colstart + [1:size(model.surface{surf2(p)}.node,2)];
        colstart = cols{p}(end);
        col = [col surf2(p)*ones(size(cols{p}))];
    end    

    EJ = zeros(rowstart,colstart);

    for p = 1:length(surf1),
        for q = 1:length(surf2),     
            fprintf(1,'Calculando transferencia de la divergencia de potencial %d y %d\n',surf1(p), surf2(q));
            EJ(rows{p},cols{q}) = CalcMatrix(model,surf1(p),surf2(q));
            fprintf(1,'\n');
         end
    end           
return

function EJ = CalcMatrix(model,surf1,surf2)

    Pts = model.surface{surf1}.node;
    Pos = model.surface{surf2}.node;
    Tri = model.surface{surf2}.face;

    NumPts = size(Pts,2);
    NumPos = size(Pos,2);
    NumTri = size(Tri,2);
    
        EJ = zeros(NumPts,NumPos);
        for k = 1:NumPts,
            W = bem_radon(Tri',Pos',Pts(:,k)'); % Mejora la integracion
%             W = bem_const(Tri',Pos',Pts(:,k)'); % Considero "r" constante
            for l = 1:NumTri,
                EJ(k,Tri(:,l)) = EJ(k,Tri(:,l)) + (1/(4*pi))*W(l,:);
            end        
        end
return

function W = bem_const(TRI, POS, OBPOS)
% Considero "r" constante en cada cara del triangulo para la integral.

nrTRI = size(TRI,1);

% esquinas, el centro y el área de cada triángulo    
P1 = POS(TRI(:,1),:); 
P2 = POS(TRI(:,2),:); 
P3 = POS(TRI(:,3),:);
C = (P1 + P2 + P3) / 3;
N = cross(P2-P1,P3-P1);
A = 0.5 * sqrt(sum(N'.^2))';

qv = C-ones(nrTRI,1)*OBPOS; 
r=sqrt(sum(qv'.^2));
W = A./r';
% keyboard
return
 
function W = bem_radon(TRI,POS,OBPOS);

% La aproximacion de la integracion se hace mediante cuadratura de Gauss:
% Orden 5, 7 puntos en el triangulo a integrar.

% "TRI" Los índices de los triángulos
% "POS" Posiciones [x, y, z] de los triángulos
% "OBPOS" Posicion del punto de observación [x, y, z]
% "W" Peso, linealmente distribuidas en triángulo.

% pesos iniciales
s15 = sqrt(15);
w1 = 9/40;
w2 = (155+s15)/1200; w3 = w2; w4 = w2;
w5 = (155-s15)/1200; w6 = w5; w7 = w6;
s  = (1-s15)/7;
r  = (1+s15)/7;

nrTRI = size(TRI,1);
nrPOS = size(POS,1);
W = zeros(nrTRI,3);

epsilon=1e-20; % Para detectar triangulos singulares %%%%%%%%%%%%%%%%%%%%%%%%%%%

% mover todas las posiciones de OBPOS como el origen
POS = POS - ones(nrPOS,1)*OBPOS;
I = find(sum(POS'.^2)<epsilon); Ising = [];

if ~isempty(I), 
    Ising  = [];
    for p = 1:length(I);
        [tx,dummy] = find(TRI==I(p)); 
        Ising = [Ising tx'];
    end
end

% esquinas, el centro y el área de cada triángulo
P1 = POS(TRI(:,1),:); 
P2 = POS(TRI(:,2),:); 
P3 = POS(TRI(:,3),:);
C = (P1 + P2 + P3) / 3;
N = cross(P2-P1,P3-P1);
A = 0.5 * sqrt(sum(N'.^2))';

% punto de suma (posiciones)
q1 = C;
q2 = s*P1 + (1-s)*C;
q3 = s*P2 + (1-s)*C;
q4 = s*P3 + (1-s)*C;
q5 = r*P1 + (1-r)*C;
q6 = r*P2 + (1-r)*C;
q7 = r*P3 + (1-r)*C;

% norma de las posiciones
espsilon2=0;
nq1 = sqrt(sum(q1'.^2))'+espsilon2;
nq2 = sqrt(sum(q2'.^2))'+espsilon2;
nq3 = sqrt(sum(q3'.^2))'+espsilon2;
nq4 = sqrt(sum(q4'.^2))'+espsilon2;
nq5 = sqrt(sum(q5'.^2))'+espsilon2;
nq6 = sqrt(sum(q6'.^2))'+espsilon2;
nq7 = sqrt(sum(q7'.^2))'+espsilon2;

% Factores de peso para la distribución lineal de los puntos fuertes
a1 = 2/3; b1 = 1/3;
a2 = 1-(2*s+1)/3; b2 = (1-s)/3;
a3 = (s+2)/3; b3 = b2;
a4 = a3; b4 = (2*s+1)/3;

a5 = 1-(2*r+1)/3; b5 = (1-r)/3;
a6 = (r+2)/3; b6 = b5;
a7 = a6; b7 = (2*r+1)/3;

% s  = (1-s15)/7;
% r  = (1+s15)/7;
% a1 = 2/3; b1 = 1/3;
% a2 = 1-(2*s+1)/3 = 1-((2-2s15)/(7*3)+1/3) = 1-((2-2s15)/21+7/21) =
%      1-(9-2s15)/21 = ((21-9)+2s15)/21 = (12+2s15)/21 = 2(6+s15)/21
% b2 = (1-s)/3 = (1-(1-s15)/7)/3 = 7/21-(1-s15)/21 = (6+s15)/21
% a3 = (s+2)/3 = ((1-s15)/7+2)/3 = (1-s15)/21+14/21 = (15-s15)/21
% b4 = (9-2s15)/21;
% b3=b2; a4=a3; 
% a5 = 1-(2*r+1)/3 = 1-(2+2s15)/21+7/21 = 1-(9+2s15)/21 = 2(6-s15)/21
% b5 = (1-r)/3 = 7/21-(1+s15)/21 = (6-s15)/21
% a6 = (r+2)/3 = (1+s15)/21+14/21 = (15+s15)/21
% b7 = (9+2s15)/21
% b6=b5; a7=a6; 

% diferentes pesos calculados
W(:,1) = A.*((1-a1)*w1./nq1 + (1-a2)*w2./nq2 + (1-a3)*w3./nq3 + (1-a4)*w4./nq4 + (1-a5)*w5./nq5 + (1-a6)*w6./nq6 + (1-a7)*w7./nq7);
W(:,2) = A.*((a1-b1)*w1./nq1 + (a2-b2)*w2./nq2 + (a3-b3)*w3./nq3 + (a4-b4)*w4./nq4 + (a5-b5)*w5./nq5 + (a6-b6)*w6./nq6 + (a7-b7)*w7./nq7);
W(:,3) = A.*(b1*w1./nq1 + b2*w2./nq2 + b3*w3./nq3 + b4*w4./nq4 + b5*w5./nq5 + b6*w6./nq6 + b7*w7./nq7);

% triangulos singulares
for i=1:length(Ising),
	I = Ising(i);
	W(I,:) = bem_sing(POS(TRI(I,:),:), epsilon);
end    
    
return    

function W = bem_sing(TRIPOS, epsilon);

% W (J) es la contribución en el vértice de una factor  unidad
% en el vértice J, J = 1,2,3

ISIN = find(sum(TRIPOS'.^2)<epsilon);
if isempty(ISIN), error('Did not find singularity!'); return; end
temp = [1 2 3;2 3 1;3 1 2];
ARRANGE = temp(ISIN,:);

%Divide los arista RA RB RC
% A nodo singular
RA = TRIPOS(ARRANGE(1),:);
RB = TRIPOS(ARRANGE(2),:);
RC = TRIPOS(ARRANGE(3),:); 

% A es el vertices observador
[RL,AP] = laline(RA,RB,RC);

% find length of vectors BC,BP,CP,AB,AC
BC = norm(RC-RB);
BP = abs(RL)*BC;
CP = abs(1-RL)*BC;
AB = norm(RB-RA);
AC = norm(RC-RA);

% establece los pesos  del triángulo rectangulo APB
% WAPB(J) es la contribución al vértice A 
% La fuerza unidad en el vértice J, J = A, B, C
if abs(RL) > epsilon,
	a = AP; 
	b = BP;
	c = AB;
	log_term = log( (b+c)/a );
	WAPB(1) = a/2 * log_term;
	w = 1-RL; 
	WAPB(2) = a* (( a-c)*(-1+w) + b*w*log_term )/(2*b);
	w = RL;
	WAPB(3) = a*w *( a-c  +  b*log_term )/(2*b);
else
	WAPB = [0 0 0];
end

% establece los pesos  del triángulo rectangulo APB
% WAPC(J) es la contribución al vértice A 
% La fuerza unidad en el vértice J, J = A, B, C
if abs(RL-1) > epsilon,
	a = AP;
	b = CP;
	c = AC;
	log_term = log( (b+c)/a );
	WAPC(1) = a/2 * log_term;
	w = 1-RL;
	WAPC(2) = a*w *( a-c  +  b*log_term )/(2*b);
	w = RL;
	WAPC(3) = a* (( a-c)*(-1+w) + b*w*log_term )/(2*b);
else
	WAPC = [ 0 0 0];
end

% Calcular el peso total, teniendo en cuenta la posición P en BC
if RL<0, WAPB = -WAPB; end
if RL>1, WAPC = -WAPC; end
W = WAPB + WAPC;

W(ARRANGE) = W;

return


function [rl,ap] = laline(ra,rb,rc);
 % encuentra la proyección P del vertice A (punto de observación) a la
 % línea BC
rba = ra - rb;
rbc = rc - rb;

% normaliza
nrbc = rbc / norm(rbc) ;
rl = rba * nrbc';
rl = rl / norm(rbc) ;
p = rl*(rbc)+rb; % punto de proyeccion

% distancia
ap = norm(ra-p);
return

















