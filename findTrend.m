%% Finding the trend variations in SW data to match CME-ICME pairs. 
% Written by: Mohamed Nedal 
function findTrend(cellNumFromArrayC) 
%% Read SW data 
load('soho_omni_matching.mat'); 
omni = C{cellNumFromArrayC}; 
B = omni(:,5); 
Bz = omni(:,8); 
T = omni(:,9); 
v = omni(:,11); 
na_np = omni(:,12); 
p = omni(:,13); 
Dst = omni(:,15); 
t = (0:119)'; 
%% Calculate Texp 
if mean(v) >= 500 
    Texp = 0.5 * (((0.031*v) - 5.1).^2) * 10^3; 
else 
    Texp = 0.5 * ((0.0106*v) - 0.278).^2; 
end 
a = diff(v)./diff(t); 
%% Estimate the arrival time of CME 
%% find sudden variations 
for i = 1:length(omni) 
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
%% PLOTTING 
figure; findchangepts(B,'MaxNumChanges',5); 
figure; findchangepts(Bz,'MaxNumChanges',5); 
figure; findchangepts(Dst,'MaxNumChanges',5); 
figure; findchangepts(T,'MaxNumChanges',5); 
figure; findchangepts(Texp,'MaxNumChanges',5); 
figure; findchangepts(v,'MaxNumChanges',5); 
figure; findchangepts(a,'MaxNumChanges',5); 
%% 
m1 = mean(idxB); 
m2 = mean(idxBz); 
m3 = mean(idxDST); 
m4 = mean(idxT); 
m5 = mean(idxTEXP); 
m6 = mean(idxV); 
m7 = mean(idxA); 
transTime = (m1+m2+m3+m4+m5+m6+m7)/7; 
fprintf('Expected Transit Time is %0.2f hours. \n\n', transTime); 
%% PLOTTING TRANSIT TIME OVER OMNI DATA 
sub1 = subplot(5, 1, 1);  
plot(t, B); 
hold on 
plot(t, Bz, 'r'); 
line([t(round(transTime)) t(round(transTime))],ylim,'LineStyle','--','LineWidth',2,'Color','k'); 
hold off 
ylabel('IMF (nT)'); 
legend('Bt','Bz'); 
xlim(sub1, [0 length(B)]); 
set(gca,'box','off'); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'Xticklabel',[]); 
set(gca,'box','off'); 
grid on 
ax = gca; 
ax.XTick = [0,24,48,72,96,119]; 

sub2 = subplot(6, 1, 2); 
yyaxis left 
plot(t, T); 
ylabel('Temp. (K)'); 
yyaxis right 
plot(t, Texp, 'r'); 
hold on 
line([t(round(transTime)) t(round(transTime))],ylim,'LineStyle','--','LineWidth',2,'Color','k'); 
hold off 
ylabel('Texp (0.5)'); 
legend('T','Texp'); 
xlim(sub2, [0 length(T)]); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'Xticklabel',[]); 
set(gca,'box','off'); 
grid on 
ax = gca; 
ax.XTick = [0,24,48,72,96,119]; 

sub3 = subplot(6, 1, 3); 
yyaxis left 
plot(t, v); 
ylabel('Vsw (km/s)'); 
yyaxis right 
plot(t(1:numel(a)), a) 
hold on 
line([t(round(transTime)) t(round(transTime))],ylim,'LineStyle','--','LineWidth',2,'Color','k'); 
hold off 
ylabel('a (km/hr^2)') 
legend('V','a'); 
xlim(sub3, [0 length(v)]); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'Xticklabel',[]); 
set(gca,'box','off'); 
grid on 
ax = gca; 
ax.XTick = [0,24,48,72,96,119]; 

y = 0.08; 
sub4 = subplot(6, 1, 4); 
plot(t, na_np); 
hold on 
plot(t, y*ones(size(t))); 
line([t(round(transTime)) t(round(transTime))],ylim,'LineStyle','--','LineWidth',2,'Color','k'); 
hold off 
ylabel('Na/Np'); 
xlim(sub4, [0 length(na_np)]); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'Xticklabel',[]); 
set(gca,'box','off'); 
grid on 
ax = gca; 
ax.XTick = [0,24,48,72,96,119]; 

sub5 = subplot(6, 1, 5); 
plot(t, p); 
hold on 
line([t(round(transTime)) t(round(transTime))],ylim,'LineStyle','--','LineWidth',2,'Color','k'); 
hold off 
ylabel('P (nPa)'); 
xlim(sub5, [0 length(p)]); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'Xticklabel',[]); 
set(gca,'box','off'); 
grid on 
ax = gca; 
ax.XTick = [0,24,48,72,96,119]; 

sub6 = subplot(6, 1, 6);
plot(t, Dst); 
hold on 
line([t(round(transTime)) t(round(transTime))],ylim,'LineStyle','--','LineWidth',2,'Color','k'); 
hold off 
xlabel('Time (days)'); 
ylabel('Dst (nT)'); 
ylim(sub6, [min(Dst) max(Dst)]); 
xlim(sub6, [0 length(Dst)]); 
set(gca,'XMinorTick','on','YMinorTick','on'); 
set(gca,'box','off'); 
grid on 
ax = gca; 
ax.XTick = [0,24,48,72,96,119]; 
ax.XTickLabel = [0,1,2,3,4,5]; 
end 