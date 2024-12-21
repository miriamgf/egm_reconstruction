function parametros_Espectrales_FV()
%funcion para extraer los parametros espectrales de la poblacion de FV

load('/home/ojki/Documentos/Oscar/Investigacion/InvestigacionAnalisisEspectralEGM/BaseDatos/Serce_Para_Paper/BaseDatos_Paper/FV/FVPoblacion/patFV.mat');

for m = 1:length(patFV) %#ok<NODEF>

    %Recorremos episodios

    for n = 1:length(patFV(m).epi)
        clc
        disp(['Pat ',num2str(m),'. Epi ',num2str(n)])
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %parametros clasicos
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         patFV(m).epi(n).DFA.fvpar_egm = parametrosFV3(patFV(m).epi(n).egm,patFV(m).epi(n).fs); %#ok<AGROW>
%         patFV(m).epi(n).DFA.fvpar_egm_aux = parametrosFV3(patFV(m).epi(n).egm_aux,patFV(m).epi(n).fs); %#ok<AGROW>
% 
%         %Reconstruccion usando f0_DFA
% 
%         n_harmonics = floor(patFV(m).epi(n).fs/2/patFV(m).epi(n).DFA.fvpar_egm.f0);
%         n_harmonics_aux = floor(patFV(m).epi(n).fs/2/patFV(m).epi(n).DFA.fvpar_egm_aux.f0);
% 
%         [patFV(m).epi(n).DFA.Modelo.Egm.y_est,e_est,patFV(m).epi(n).DFA.Modelo.Egm.probs,...
%             patFV(m).epi(n).DFA.Modelo.Egm.y_partial,patFV(m).epi(n).DFA.Modelo.Egm.amplis]...
%             = myHarmonicSeries(patFV(m).epi(n).DFA.fvpar_egm.egm,patFV(m).epi(n).fs,n_harmonics,patFV(m).epi(n).DFA.fvpar_egm.f0); %#ok<AGROW,AGROW>
%         [patFV(m).epi(n).DFA.Modelo.Egm_aux.y_est,e_est,patFV(m).epi(n).DFA.Modelo.Egm_aux.probs,...
%             patFV(m).epi(n).DFA.Modelo.Egm_aux.y_partial,patFV(m).epi(n).DFA.Modelo.Egm_aux.amplis]...
%             = myHarmonicSeries(patFV(m).epi(n).DFA.fvpar_egm_aux.egm,patFV(m).epi(n).fs,n_harmonics_aux,patFV(m).epi(n).DFA.fvpar_egm_aux.f0); %#ok<AGROW>
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %FOA
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Flags.FV = 1;
        Flags.RS = 0;
        [egm_FV,egm_aux_FV] = preProcesadoFV(patFV(m).epi(n));
        [patFV(m).epi(n).FOA.Egm.f0] = Estimacion_Freq_Fundamental(...
          egm_FV,[],patFV(m).epi(n).fs,Flags);
        [patFV(m).epi(n).FOA.Egm_aux.f0] = Estimacion_Freq_Fundamental(...
            egm_aux_FV,[],patFV(m).epi(n).fs,Flags);
        
        n_harmonics = floor(patFV(m).epi(n).fs/2/patFV(m).epi(n).FOA.Egm.f0);
        n_harmonics_aux = floor(patFV(m).epi(n).fs/2/patFV(m).epi(n).FOA.Egm_aux.f0);

        [patFV(m).epi(n).FOA.Modelo.Egm.y_est,e_est,patFV(m).epi(n).FOA.Modelo.Egm.probs,...
            patFV(m).epi(n).FOA.Modelo.Egm.y_partial,patFV(m).epi(n).FOA.Modelo.Egm.amplis]...
            = myHarmonicSeries(patFV(m).epi(n).DFA.fvpar_egm.egm,patFV(m).epi(n).fs,n_harmonics,patFV(m).epi(n).FOA.Egm.f0); %#ok<AGROW,AGROW>
        
        [patFV(m).epi(n).FOA.Modelo.Egm_aux.y_est,e_est,patFV(m).epi(n).FOA.Modelo.Egm_aux.probs,...
            patFV(m).epi(n).FOA.Modelo.Egm_aux.y_partial,patFV(m).epi(n).FOA.Modelo.Egm_aux.amplis]...
            = myHarmonicSeries(patFV(m).epi(n).DFA.fvpar_egm_aux.egm,patFV(m).epi(n).fs,n_harmonics_aux,patFV(m).epi(n).FOA.Egm_aux.f0); %#ok<AGROW>

    end
end

%save('patFV','patFV')
keyboard

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Funciones auxiliares%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [egm,egm_aux] = preProcesadoFV(pat)

%%Acondicionamiento señales de FV

egm = pat.DFA.fvpar_egm.egm;
egm_aux = pat.DFA.fvpar_egm_aux.egm;


%Suavizado
% [indmax,indmin] = maxmin(egm);
% [indmax_a,indmin_a] = maxmin(egm_aux);
% [envUpper,envLower] = envelope(indmax,indmin,egm,t);
% [envUpper_a,envLower_a] = envelope(indmax_a,indmin_a,egm_aux,t_aux);
% egm = (envLower + envUpper)/2;
% egm_aux = (envLower_a + envUpper_a)/2;

%Filtrado
b = fir1(6,[3/pat.fs 20/pat.fs]); %TO_DO probar con filtrado en una banda 3-20 Hz
egm = filter(b,1,egm);
egm_aux = filter(b,1,egm_aux);
end


