%% setup
clear; clc;close all;
addpath(genpath(pwd));
% rng('default');


%% BP
load('/home/wangshuang427/matlabcode/model/bp_tooth_2.mat', 'v1000')
v=v1000/1e3;
% 
v0=(min(v(:))+max(v(:)))/2.0*ones(size(v));%2.2
v0 = repmat({v0}, 12, 1);
down=1500*ones(size(v));up=2600*ones(size(v)); % 1.5/2.1
%%
% v0 = (down + (up - down) .* rand(size(down)))/1e3; %random model
% v0=(down+up)/2e3; % gradient model
dz = 25;
dx = 25;  % bp-20/mar-25
f=2:1:19; % bp15/mar19
% wavelet
f0=8;t0=0;
% wavelet
wavelet = fwi_wavelet(f,t0,f0); % ricker

xr=((2:2:size(v,2))-1)*dx;
zr = (ones(1,length(xr))+0)*dz;
xs=((3:8:size(v,2))-1)*dx;
zs = (ones(1,length(xs))+0)*dz;
%% 观测系统
n  = size(v);
h  = [dz dx];
z  = (0:n(1)-1)*h(1);
x  = (0:n(2)-1)*h(2);
Pr = getP(h,n,zr,xr);
Ps = getP(h,n,zs,xs);
mask=ones(size(v));
mask(1:5,:)=0;
mask(end,:)=0;mask(:,[1,end])=0;

for i=1:12
    v0{i}(mask==0)=v(mask==0);
    m0{i}=v0{i}(:);
end
down=down(:);up=up(:);
opt.down=down(mask==1)/1e3;
opt.up=up(mask==1)/1e3;
% v0(mask==0)=v(mask==0);
mref=v(:);
%% 模型参数
model.mref=mref;
model.f = f;
model.n = n;
model.h = h;
model.zr = zr;
model.xr = xr;
model.zs = zs;
model.xs = xs;
model.mref = mref;
model.Pr=Pr;
model.wavelet=wavelet;
model.mask=mask(:);

%% 模型先验
priori.pm1=m0{1};
Cm=10^2;
priori.pm2=Cm;
priori.lower_bounds=-inf; 
priori.upper_bounds=inf;
priori.method='Normal';
delete(gcp('nocreate'));
numWorkers=3;
% 启动并行池
% parpool('Processes', numWorkers);
parpool('Local_1', 12);

[Dt]  = F_v(mref,Ps,model);

% Adding noise to data (default noise level = 0.01)
NoiseLevel = 0.06;
for ii=1:size(Dt,3)
    [Dt_noise(:,:,ii), NoiseInfo] = PRnoise(Dt(:,:,ii),NoiseLevel);
end
% Data covariance Cd*eye(n)
Cd = NoiseLevel; %(NoiseInfo.snr*NoiseInfo.level);
Cdi = 1./Cd^2;
%% Initiate SGLD samples
N = 1e5;                      % Number of samples
ns = 1;                       % Number of samplers
for i=1:12
    x0{i} = m0{i}(mask==1);     % Initial point
end
xi = length(x0);                % Size of random vector

%% FWI
funh = @(m)misfit_vfwi_mask(m,Dt_noise,Ps,model,Cdi,priori);
% funh = @(m)misfit_vfwi_v3(m,Dt_noise,Ps,model,Cdi,priori);

% 检查当前并行池的 worker 数量
pool = gcp;
disp(['当前并行池的 worker 数量: ', num2str(pool.NumWorkers)]);

option = cell(numWorkers, 1);
step_size=1.0e-3 *ones(numWorkers,1);
gama=1*ones(numWorkers,1);
zeta=1*ones(numWorkers,1);%bp-1
beta=1 *ones(numWorkers,1);
M=1;P=1;

nchain=1;
X = zeros(length(x0{1}),N,nchain);               % Samples
fk = zeros(N+1,nchain);                          % Samples

% ii=1;
tic;
for ii=1:nchain

    [X(:,:,ii),fk(:,ii)] = sampler_ThirdOrder_Precond_UULA(funh,N,x0{2},step_size(ii),gama(ii),zeta(ii),beta(ii),M,P,opt);

end
elapsedTime=toc;
% figure;plot(fk);title('misfit');


