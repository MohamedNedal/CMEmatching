function transTime = cme_icme_V4(omniMATRIX, Vcme) 
%% This fucntion is for match CME-ICME pairs. 
%% Written by: Mohamed Nedal 
% Thanks to Stefan Lotz (SANSA) for his suggestions. 
%% Read IP data 
B = omniMATRIX(:,5); 
% Bx = omniMATRIX(:,6); 
% By = omniMATRIX(:,7); 
Bz = omniMATRIX(:,8); 
T = omniMATRIX(:,9); 
% np = omniMATRIX(:,10); 
v = omniMATRIX(:,11); 
na_np = omniMATRIX(:,12); 
p = omniMATRIX(:,13); 
% beta = omniMATRIX(:,14); 
Dst = omniMATRIX(:,15); 
% Define Time on X-axis 
t = (0:119)';               % 5 days 
%% Fill with NaN for false values 
Bz(Bz == 9999.99000000000) = NaN; 
v(v == 99999.9000000000) = NaN; 
T(T == 9999999) = NaN; 
na_np(na_np == 9.99900000000000) = NaN; 
% Calculate Texp 
if mean(v) >= 500 
    Texp = 0.5 * (((0.031*v) - 5.1).^2) * 10^3; 
else 
    Texp = 0.5 * ((0.0106*v) - 0.278).^2; 
end 
a = diff(v)./diff(t); 
%% Find Sudden Variations 
for i = 1:length(omniMATRIX) 
    if isnan(B(i)) 
        B = smooth(B,0.01,'loess'); 
    elseif isnan(Bz(i)) 
        Bz = smooth(Bz,0.01,'loess'); 
    elseif isnan(Dst(i)) 
        Dst = smooth(Dst,0.01,'loess'); 
    elseif isnan(T(i)) 
        T = smooth(T,0.01,'loess'); 
    elseif isnan(Texp(i)) 
        Texp = smooth(Texp,0.01,'loess'); 
    elseif isnan(v(i)) 
        v = smooth(v,0.01,'loess'); 
    end 
end 
clear i; 
idxB = findchangepts(B,'MaxNumChanges',5); 
idxBz = findchangepts(Bz,'MaxNumChanges',5); 
idxDST = findchangepts(Dst,'MaxNumChanges',5); 
idxT = findchangepts(T,'MaxNumChanges',5); 
idxTEXP = findchangepts(Texp,'MaxNumChanges',5); 
idxV = findchangepts(v,'MaxNumChanges',5); 
idxA = findchangepts(a,'MaxNumChanges',5); 
m1 = mean(idxB); 
m2 = mean(idxBz); 
m3 = mean(idxDST); 
m4 = mean(idxT); 
m5 = mean(idxTEXP); 
m6 = mean(idxV); 
m7 = mean(idxA); 
transTime = (m1+m2+m3+m4+m5+m6+m7)/7; 
fprintf('Expected Transit Time is %0.2f hours. \n', transTime); 
%% THE KINEMATIC EQUATIONS SECTION 
% Define Constants 
AU = 149599999.99979659915; % in KM 
d = 0.76 * AU; % acceleration cessation distance 
% G2001 Model 
a_calc = power(-10,-3) * ((0.0054*Vcme) - 2.2); % a => km/s^2 
squareRoot = sqrt(power(Vcme,2) + (2*a_calc*d)); 
A = (-Vcme + squareRoot) / a_calc; 
B = (AU - d) / squareRoot; 
t_hrs_G2001 = (A + B) / 3600; 
% 11.04 hrs is the avg. error of G2001 model 
% the following limits are the searching window in SW data 
upperLimit = t_hrs_G2001 + 5.52; 
lowerLimit = t_hrs_G2001 - 5.52; 
fprintf('Exp. transit time via G2001 model\n is eaither %0.2f or %0.2f hours. \n',upperLimit,lowerLimit); 
end 