%%%% Attitude estimation using EKF

clc
clear all;
close all;
load('D1.mat');

t = 0; 
datasize = size(ax,1)-t
ax = ax(t+1:t+s);
ay = ay(t+1:t+s);
az = az(t+1:t+s);
p = p(t+1:t+s);
q = q(t+1:t+s);
r = r(t+1:t+s);
mx = mx(t+1:t+s);
my = my(t+1:t+s);
mz = mz(t+1:t+s);
phi = phi(t+1:t+s);
theta = theta(t+1:t+s);
psi = psi(t+1:t+s);

phiA = zeros(1,datasize);
phiG = zeros(1,datasize);
phiG_dot = zeros(1,datasize);

thetaA = zeros(1,datasize);
thetaG = zeros(1,datasize);
thetaG_dot = zeros(1,datasize);

psiM = zeros(1,datasize);
psiG = zeros(1,datasize);
psiG_dot = zeros(1,datasize);

phiE = zeros(1,datasize);
thetaE = zeros(1,datasize);
psiE = zeros(1,datasize);

e1 = zeros(datasize,1);
e2 = zeros(datasize,1);
e3 = zeros(datasize,1);
x = zeros(3,1);
x_predict = zeros(3,datasize);
F = zeros(3);
covP_dot = zeros(3);
covP = 0.1*[1 0 0;
            0 1 0;
            0 0 1];
H = [1 0 0;
     0 1 0;
     0 0 1];
 
Q = [5 0 0;
     0 0.05 0;
     0 0 10];

R = [0.1 0 0;
     0 0.001 0;
     0 0 .8];
 
I = 0.00*H; 
m = 2;
tic
while m <= datasize
%% Model
% Process model
    phiG_dot(m) = (p(m)) + (q(m))*sin(phiE(m-1))*tan(thetaE(m-1)) + (r(m))*cos(phiE(m-1))*tan(thetaE(m-1));
    thetaG_dot(m) = (q(m))*cos(phiE(m-1)) - (r(m))*sin(phiE(m-1));
    psiG_dot(m) = (q(m))*sin(phiE(m-1))*sec(thetaE(m-1)) + (r(m))*cos(phiE(m-1))*sec(thetaE(m-1));
% Measurement model
    phiA(m) = atan(ay(m)/az(m));
    thetaA(m)= atan((-ax(m))/(ay(m)*sin(phiE(m-1))+ az(m)*cos(phiE(m-1))));
    xh = mx(m)*cos(phiE(m-1)) + my(m)*sin(thetaE(m-1))*sin(phiE(m-1)) + mz(m)*sin(thetaE(m-1))*cos(phiE(m-1));
    yh = -my(m)*cos(phiE(m-1))+ mz(m)*sin(phiE(m-1));
    psiM(m) = atan2(yh,xh);
    
    x_measure = [phiA(m); thetaA(m); psiM(m)];
%% Prediction
%   1. State estimation

    x_dot = [phiG_dot(m);
             thetaG_dot(m);
             psiG_dot(m)];
    
    x_predict = x + x_dot*dt;

%   2. Error Covariance estimation
    F = jacobianmatrix(p(m),q(m),r(m),phiE(m-1),thetaE(m-1),psiE(m-1));
    
    P_predict = F*covP*transpose(F)+Q;
%     P_predict = covP + covP_dot*dt;
%% Correction
%   1. Kalman Gain
    gainK = P_predict*transpose(H)*inv(H*P_predict*transpose(H)+R);

%   2. State Estimation
    x = x_predict + gainK*(x_measure - x_predict);
    
%   3. Error Covariance estimation
    covP = (eye(3) - gainK*H)*P_predict;
    
    e1(m) = covP(1,1);
    e2(m) = covP(2,2);
    e3(m) = covP(3,3);
%%
    phiE(m) = transpose(x(1,:));
    thetaE(m) = transpose(x(2,:));
    psiE(m) = transpose(x(3,:));
m = m+1;
end
toc
rmsephi = rmse(datasize,phi,phiE);
rmsetheta = rmse(datasize,theta,thetaE);
rmsepsi = rmse(datasize,psi,psiE);

figure(2);
subplot(3,1,1);
plot(phi,'r');
hold on;
plot(phiE,'--','color','b');
title('Roll Estimates','fontweight','bold','fontsize',12);
ylabel('\phi angle (in radian)');
legend('\phi_{ideal}','\phi_{estimated}');

subplot(3,1,2);
plot(theta,'r');
hold on;
plot(thetaE,'--','color','b');
ylabel('\theta angle (in radian)');
title('Pitch Estimates','fontweight','bold','fontsize',12);

subplot(3,1,3);
plot(psi,'r');
hold on;
plot(psiE,'--','color','b');
title('Yaw Estimates','fontweight','bold','fontsize',12);
xlabel('time');
ylabel('\psi angle (in radian)');