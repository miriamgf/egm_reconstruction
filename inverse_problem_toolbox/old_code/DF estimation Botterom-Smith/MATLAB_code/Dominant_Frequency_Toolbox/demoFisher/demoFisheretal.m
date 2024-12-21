function demoFisheretal

%

rand('seed',0); 
randn('seed',0);

while 1
    opt = menu('Choose an option:',...
        '1) See EGM(t) and spectra',...
        '2) See EGM(t) and spectra with D',...
        '3) Simmulate RI with D',...
        '4) See results',...
        '5) Simmulate RI with D / ratio correction',...
        '6) See results',...
        '7) Simmulate RI with D / harmonic correction',...
        '8) See results',...
        '9) Simmulate RI with D / single harmonic correction',...
        '10) See results',...
        '11) See EGM(t) and spectra with multicomponent',...
        '12) Bye');
%        'Simmulate RI with multicomponent',...
%        'See results',...
%        'Simmulate RI with multicomponent / harmonic correction',...
%        'See results',...
    switch opt
        case 1  % See EGM(t) and spectra

            prompt={'Input fundamental freq:'};
            def={'5'};
            dlgTitle='See EGM(t) and spectra';
            answer=inputdlg(prompt,dlgTitle,1,def);
            f0 = str2num(answer{1});
            [egm,t] = pseudoegm(f0);
            
            figure(1)
            drawegm(egm,t);
            figure(2)
            drawegm(abs(egm),t);
            
        case 2  % See EGM(t) and spectra with D
         
            prompt={'Input fundamental freq:','Cycle uncertainty D (ms}'};
            def = {'5','5'};
            dlgTitle='See EGM(t) and spectra';
            answer=inputdlg(prompt,dlgTitle,1,def);
            f0 = str2num(answer{1});
            D = str2num(answer{2});
            [egm,t] = pseudoegm(f0,D);
            
            figure(1)
            drawegm(egm,t);
            figure(2)
            drawegm(abs(egm),t);
            
        case 3  % Simmulate RI with D
            demoriwithD;
            
        case 4  % See results
            load resultDemowithD 
            figure(3),clf
            surf(D,f0,RItot), axis tight, shading interp
            ejes = axis; ejes(5:6) = [0 1]; axis(ejes)
            xlabel('Uncertainty D (ms)');  ylabel('f0 (Hz)');  zlabel('RI')

        case 5  % Simmulate RI with D / ratio correction
            demoriwithDratioCorrection

        case 6  % See results
            load resultDemowithDrationcorrection

            figure(4),clf
            surf(D,f0,RItot), axis tight, shading interp
            ejes = axis;
            ejes(5:6) = [0 1]; axis(ejes)
            xlabel('Uncertainty D (ms)')
            ylabel('f0 (Hz)')
            zlabel('RI')
            title('Theoretical ratio correction')

        case 7  % Simmulate RI with D / harmonic correction
            demoriwithDharmonicCorrection;
            
        case 8  % See results
            load resultDemowithDHC

            figure(5),clf
            surf(D,f0,RItot), axis tight, shading interp
            ejes = axis;
            ejes(5:6) = [0 1]; axis(ejes)
            xlabel('Uncertainty D (ms)')
            ylabel('f0 (Hz)')
            zlabel('RI')
            title('Harmonic correction')

        case 9      % Simmulate RI with D / single harmonic correction
            demoriwithDsingleharmonicCorrection;
            
        case 10     % See results
            load resultDemoSingleHarmonic;
            figure(6),clf
            surf(D,f0,RItot), axis tight, shading interp
            ejes = axis;
            ejes(5:6) = [0 1]; axis(ejes)
            xlabel('Uncertainty D (ms)')
            ylabel('f0 (Hz)')
            zlabel('RI')
            title('Single Harmonic correction')

        case 11  % See EGM(t) and spectra with multicomponent
          %pseudo_multicomponent;
          
            disp('Work in progress');
         case 10  % Simmulate RI with multicomponent
%             disp('Work in progress');
%         case 11  % See results
%             disp('Work in progress');
%         case 12  % Simmulate RI with multicomponent / harmonic correction
%             disp('Work in progress');
%         case 13  % See results
%             disp('Work in progress');
        case 12  % Bye
            break;
    end
end