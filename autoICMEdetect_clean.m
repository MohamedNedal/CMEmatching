%% This code for plotting IP parameters to detect shocks & ICMEs 
% the green lines lindicate a potential signature of ICMEs 
% the red line indicates the arrival time of ICME 
% that caused a geomagnetic storm 
%% Convert this code to a function and loop it on a list of CMEs. 
% Create a matrix with the CMEs in the 1st column 
% and the 5-days period after it in SW data, as a structure object. 
%% For each CME, plot the height-time profile on left-side & SW indicies on right-side. 
%% Written by: Mohamed Nedal 
% Thanks to Stefan Lotz (SANSA) for his suggestions. 
close all; clear; clc 
% 4  Field Magnitude, nT 
% 5  BX, GSE, nT 
% 6  BY, GSM, nT 
% 7  BZ, GSM, nT 
% 8  Temperature, K 
% 9  Proton Density, n/cc 
% 10 Speed, km/s 
% 11 Alpha/Prot. ratio 
% 12 Flow pressure, nPa 
% 13 Plasma betta 
% 14 SYM/H (Dst), nT 

%% Read IP data 
omni = load('omni.txt'); 
B = omni(:,4); 
Bz = omni(:,7); 
T = omni(:,8); 
v = omni(:,10); 
np = omni(:,10); 
na_np = omni(:,11); 
h = 0.08; 
p = omni(:,12); 
pbeta = omni(:,13); 
Dst = omni(:,14); 
% Define Time on X-axis 
t = (0:length(omni)-1)'; 

%% Fill with NaN for false values 
Bz(Bz == 9999.99000000000) = NaN; 
v(v == 99999.9000000000) = NaN; 
T(T == 9999999) = NaN; 
na_np(na_np == 9.99900000000000) = NaN; 

%% Calculate Texp & plot it with Tp 
if mean(v) >= 500 
    % Texp1_1 = ((0.77 + 0.021)*v) - (265 + 12.5); 
    % Texp1_2 = ((0.77 - 0.021)*v) - (265 - 12.5); 
    Texp = 0.5 * (((0.031*v) - 5.1).^2) * 10^3; 

else 
    Texp = ((0.0106*v) - 0.278).^2;
    % Texp2_1 = ((0.0106 + 0.001)*v) - (0.278 + 0.03); 
    % Texp2_2 = ((0.0106 - 0.001)*v) - (0.278 - 0.03); 
end 

%% Plot Texp & Tp 
clc; figure 
[val, idx] = min(Dst); % "idx" is the moment of the Min. Dst 

% Plot Bz with time 
Bmnt = mean(B); 
Bmnz = mean(Bz); 
sub1 = subplot(5, 1, 1);  
plot(t, B)
hold on 
plot(t, Bz, 'r')
plot(t, Bmnt*ones(size(t)), 'k')
plot(t, Bmnz*ones(size(t)), 'c')
line([t(idx) t(idx)], ylim, 'LineStyle','--', 'Color', 'r')
[rowB, ~] = find (B > Bmnt & Bz < Bmnz); 
line([t(rowB) t(rowB)], ylim, 'LineStyle','--', 'Color', 'g')
title('SW parameters observed by OMNI at 1 AU')
ylabel('IMF (nT)')
%ymin = min(Bz); ymax = max(B); 
%ylim(sub1, [ymin ymax]); 
lenB = length(B); 
xlim(sub1, [0 lenB]); 
legend('Bt', 'Bz', 'mean(B)', 'mean(Bz)'); 
set(gca,'box','off'); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'Xticklabel',[]); 
set(gca,'box','off'); 

% Plot T & Texp with time 
sub2 = subplot(6, 1, 2); 
yyaxis left 
plot(t, T); 
ylabel('Temp. (K)')
[row1, ~] = find (Texp > T); 
yyaxis right 
plot(t, Texp, 'r')
hold on 
line([t(row1) t(row1)], ylim, 'LineStyle','--', 'Color', 'g')
line([t(idx) t(idx)], ylim, 'LineStyle','--', 'Color', 'r')
ylabel('Texp (0.5)')
xlim(sub2, [0 length(T)])
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'Xticklabel',[])
set(gca,'box','off')

% Plot Vsw with time 
sub3 = subplot(6, 1, 3); 
yyaxis left 
plot(t, v); 
hold on 
line([t(idx) t(idx)], ylim, 'LineStyle','--', 'Color', 'r')
ylabel('Vsw (km/s)')
a = diff(v)./diff(t); 
yyaxis right 
plot(t(1:numel(a)), a) 
ylabel('a (km/hr^2)') 
xlim(sub3, [0 length(v)])
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'Xticklabel',[])
set(gca,'box','off')

% Plot Nalpha/Np ratio with time 
sub4 = subplot(6, 1, 4); 
y = 0.08; 
plot(t, na_np) 
hold on 
plot(t, y*ones(size(t)))
[row2, ~] = find (na_np > 0.08); 
line([t(row2) t(row2)], ylim, 'LineStyle','--', 'Color', 'g')
line([t(idx) t(idx)], ylim, 'LineStyle','--', 'Color', 'r')
ylabel('Na/Np')
legend('Na/Np', 'Baseline (0.08)')
lenN = length(na_np); 
xlim(sub4, [0 lenN])
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'Xticklabel',[])
set(gca,'box','off')

% Plot SW Pressure with time 
sub5 = subplot(6, 1, 5); 
plot(t, p)
hold on 
line([t(idx) t(idx)], ylim, 'LineStyle', '--', 'Color', 'r')
ylabel('P (nPa)')
lenP = length(p); 
xlim(sub5, [0 lenP])
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'Xticklabel',[])
set(gca,'box','off')

% Plot Dst with time 
sub6 = subplot(6, 1, 6); 
plot(t, Dst)
hold on 
[row3, ~] = find (Dst < -50); 
line([t(row3) t(row3)], ylim, 'LineStyle','--', 'Color', 'g')
line([t(idx) t(idx)], ylim, 'LineStyle','--', 'Color', 'r')
est_arrival = (row1(1) + row2(1) + rowB(1) + idx)/4; 
line([t(round(est_arrival)) t(round(est_arrival))], ylim, 'LineStyle','--', 'Color', 'k')
fprintf('Est. Arrival Time is at the hour: %s \n', num2str(est_arrival))
xlabel('Time (days)')
ylabel('Dst (nT)')
ymin = min(Dst); ymax = max(Dst); 
ylim(sub6, [ymin ymax])
lenDst = length(Dst); 
xlim(sub6, [0 lenDst]) 
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'box','off')
legend('Dst', 'Disturbance')

% Save Fig. 
fig = gcf; 
fig.PaperUnits = 'centimeters'; 
fig.PaperPosition = [0 0 50 25]; 
fig.PaperPositionMode = 'manual'; 
figTitle = 'SW_parameters'; 
print(figTitle,'-dpng','-r0')

%% 
clc 
figure 
plot(t, B, 'b')
hold on 
plot(t, Bz, 'r')
plot(t, Bmnt*ones(size(t)), 'm')
plot(t, Bmnz*ones(size(t)), 'c')
[rowB, ~] = find (B > Bmnt & Bz < Bmnz); 
line([t(rowB) t(rowB)], ylim, 'LineStyle','--', 'Color', 'g')
est_arrival = (row1(1) + row2(1) + rowB(1) + idx)/4; 
line([t(round(est_arrival)) t(round(est_arrival))], ylim, 'LineStyle','--', 'Color', 'k')
fprintf('Est. Arrival Time is at the hour: %s \n', num2str(est_arrival))
lenB = length(B); 
xlim([0 lenB])
title('IMF components observed by OMNI at 1 AU')
ylabel('IMF (nT)')
xlabel('Time (hours)')
legend('Bt', 'Bz', 'mean(Bt)', 'mean(Bz)', 'Disturbance')
set(gca,'box','off')

% Save Fig. 
fig = gcf; 
fig.PaperUnits = 'centimeters'; 
fig.PaperPosition = [0 0 40 20]; 
fig.PaperPositionMode = 'manual'; 
figTitle = 'IMF_components'; 
print(figTitle,'-dpng','-r0') 

%% THE KINEMATIC EQUATIONS SECTION 
% soho = load('...........'); 
% soho_cme_time = soho(:,...); 
% Vcme = soho(:,...)' 

Vcme = 800; % just as a test ... 
soho_cme_time = 1; % just as a test ... 

% Define Constants 
fprintf('Define constants & empty matrices ... \n'); 

% AU => km 
AU = 149599999.99979659915; 

% acceleration cessation distance 
d = 0.76 * AU; 
t_hrs_G2001 = zeros(size(Vcme)); 
fprintf('Prediction using Kinematic Equation ... \n'); 

% G2001 Model 
fprintf('G2001 Model ... \n'); 
% a => km/s^2 
a_calc(soho_cme_time) = power(-10,-3) * ((0.0054*Vcme(soho_cme_time)) - 2.2); 
squareRoot(soho_cme_time) = sqrt(power(Vcme(soho_cme_time),2) + (2*a_calc(soho_cme_time)*d)); 
A(soho_cme_time) = (-Vcme(soho_cme_time) + squareRoot(soho_cme_time)) / a_calc(soho_cme_time); 
B(soho_cme_time) = (AU - d) / squareRoot(soho_cme_time); 
t_hrs_G2001(soho_cme_time) = ( A(soho_cme_time) + B(soho_cme_time) ) / 3600; 
% 11.04 hrs is the avg. error of G2001 model 

% the following limits are the searching window in SW data 
upperLimit = t_hrs_G2001 + 5.52; 
lowerLimit = t_hrs_G2001 - 5.52; 
fprintf('Expected transit time based on G2001 model\nis eaither %0.2f hours or %0.2f hours.\n', upperLimit, lowerLimit)

%% 
% close all; clear; clc 
% %% read omni data 
% omni_data = load('omni2013.txt'); 
% omni_datetime = datetime(2013,1,omni_data(:,2),omni_data(:,3),0,0); 
% omni = table(omni_datetime, omni_data(:,4:end)); 
% omni = [omni(:,1), array2table(omni.Var2,'VariableNames',...
%     {'Bt','Bx','By','Bz','T','n','v','alpha_p_ratio','p','beta','dst'})]; 
% % date 
% omni_datetime.Format = 'dd-MM-yyyy'; 
% omni2 = table(omni_datetime, omni_data(:,4:end)); 
% omni2 = [omni2(:,1), array2table(omni2.Var2,'VariableNames',...
%     {'Bt','Bx','By','Bz','T','n','v','alpha_p_ratio','p','beta','dst'})]; 
% 
% %% read soho data 
% soho_data = readtable('soho2013.xlsx'); 
% dSOHO = soho_data(:,{'datetimeC2'}); 
% % date 
% soho_date = dSOHO.datetimeC2; 
% soho_date.Format = 'dd-MM-yyyy'; 
% soho2 = table(soho_date, soho_data{:,3:end}); 
% soho2 = [soho2(:,1), array2table(soho2.Var2,'VariableNames',...
%     {'w','vl','vf','v20R','a','MPA'})]; 
% % 
% soho_date = cellstr(soho_date);    % Cell array of strings. 
% % time (there's a problem with the format) 
% soho_time = dSOHO.datetimeC2; 
% soho_time.Format = 'hh:mm:ss'; 
% soho_t = cellstr(soho_time);    
% soho_time = datetime(soho_t,'Format','HH:mm:ss'); 
% 
% %% Match SOHO with OMNI2 
% % soho & omni 
% clear; clc 
% % 
% omni_data = load('omni2013.txt'); 
% omni_datetime = datetime(2013,1,omni_data(:,2),omni_data(:,3),0,0); % from DOY to datetime 
% omniVEC_datetime = datevec(omni_datetime); 
% omni = [omniVEC_datetime omni_data(:,4:end)]; 
% % 
% soho_data = xlsread('soho2013.xlsx',1); 
% soho_datetime = datetime(soho_data(:,1),'ConvertFrom','excel'); 
% sohoVEC_datetime = datevec(soho_datetime); 
% soho_data(:,1) = []; 
% soho = [sohoVEC_datetime soho_data]; 
% %% remove last rows at the end of December 
% % soho(22:23,:) = []; 
% %% 
% % to find the matched date and store the that day along with 
% % the following 5 days (120 hours) in another matrix. 
% for n = 1:size(soho,1) 
%     omniRowInd = find(omni(:,1)==soho(n,1) & omni(:,2)==soho(n,2) & omni(:,3)==soho(n,3),1,'first'); 
%     if ~isempty(omniRowInd) 
%         tempTable = omni(omniRowInd:min(omniRowInd+119,size(omni,1)),:); 
%         eval(['omniSUB_' num2str(n) '=' 'tempTable;']); 
%     end 
% end 
% %% Another Method, simplear & neater than the previous one ... 
% N = size(soho,1); 
% C = cell(1,N); 
% for k = 1:N 
%     omniRowInd = find(omni(:,1)==soho(k,1) & omni(:,2)==soho(k,2) & omni(:,3)==soho(k,3),1,'first'); 
%     if ~isempty(omniRowInd) 
%         C{k} = omni(omniRowInd:min(omniRowInd+119,size(omni,1)),:); 
%     end 
% end 
% clear k N; 
% %% Check which of those submatrices have been created or not 
% check_Exist = cellfun(@isempty, C); 
% %% Proceeding further analysis 
% T = nan(3,1); 
% sizeM = nan(numel(C),2);
% for k = 1:numel(C) 
%     M = C{k}; 
%     sizeM(k,:) = size(M); 
%     ... do whatever analysis with M 
%     T(k,:) = cme_icme_V3_noPlot(M, soho(k,8)); 
% end 
% clear k; 
% %% check correlation betn Vsoho vs T 
% % scatter(soho(:,8),T,'filled'); 
% % xlabel('CME Speed (km/s)'); 
% % ylabel('Transit Time (hr)'); 
% 
% %% Download All CME Data with OMNI Data 







