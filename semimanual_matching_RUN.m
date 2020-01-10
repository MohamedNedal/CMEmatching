%% CME-ICME Matching 
% Written by: Mohamed Nedal 
close all; clear; clc 
%% Import OMNI  
diary command_window.txt 
omni_data = load('omni_Example.txt'); 
y = omni_data(:,1); % year 
doy = omni_data(:,2); 
omniVEC_datetime = datevec(datenum(y,1,doy)); 
omniVEC_datetime(:,4:6) = []; 
omni_data(:,1:2) = []; 
omni = [omniVEC_datetime omni_data]; 
%% Import SOHO 
soho_data = xlsread('SOHOdata.xlsx',1); 
datetimeSOHO = soho_data(:,1) + soho_data(:,2); 
soho_datetime = datetime(datetimeSOHO,'ConvertFrom','excel','Format','dd/MM/yyyy HH:mm:ss'); 
sohoVEC_datetime = datevec(soho_datetime); 
soho_data(:,1:2) = []; 
soho = [sohoVEC_datetime soho_data]; 
%% Match SOHO with OMNI2 || soho & omni matrices 
% to find the matched date and store that day along with 
% the following 5 days (120 hours) in another matrix. 
N = size(soho,1); 
C = cell(1,N); 
for k = 1:N 
    omniRowInd = find(omni(:,1)==soho(k,1) & omni(:,2)==soho(k,2) & omni(:,3)==soho(k,3),1,'first'); 
    if ~isempty(omniRowInd) 
        C{k} = omni(omniRowInd:min(omniRowInd+119,size(omni,1)),:); 
    end 
    % Check which of those submatrices have been created or not 
    if ~any(C{k}) 
        fprintf('WARNING, Empty cell ... \n\n'); 
    end 
end 
clear k N; 
%% Proceeding further analysis 
Tr = nan(length(C),1); 
for k = 1:length(C) 
    M = C{k}; 
    Tr(k,:) = cme_icme_V4(M, soho(k,8)); 
    fprintf('Event #%d is done ... \n\n', k); 
end 
clear k; 
%% check correlation betn Vsoho vs T 
scatter(soho(:,8), Tr, 'filled'); 
xlabel('CME Speed (km/s)'); 
ylabel('Transit Time (hr)'); 
diary off 