function [X,f] = sampler_ThirdOrder_Precond_UULA(funh,N,x0,stepsizek,gama,zeta,beta,M,P,opt)
%  SAMPLER_THIRDORDER_PRECOND_UULA 3rd order Pre-conditioned  Underdamped Langevin dynamics with BACOCAB scheme 
% discretization.
% 
% Reference: Pierre Monmarch. (2021) 'Solving Linear Inverse Problems UsingHigher-Order 
% Annealed Langevin Diffusion'
%   d(xt)=P*inv(M)*vt*d(t)
%   d(vt)= -P*gridient*d(t)+gama*zt*d(t)
%   d(zt)= -gama*vt*d(t)-zeta*zt*d(t)+sqrt(2*zeta*beta*M)*d(Bt)
%   Implemented by  : Shuang Wang, JLU
%   Version         : Dec 11, 2024
%
%   Input:
%   N         - Number of samples
%   x0        - Initial point, vector of dimension-by-one
%   stepsize      - Set of initial step-length
%   beta      - Temperature parameters/0.3 is best LPIPS
%   gama      - Damping coefficient (friction)/ 1.5 para spacing 25
%   zeta      - Damping coefficient (friction)
%   M         - mass matrix that controls the coupling between xtand vt,
%   P         - pre-conditioned matrix/symmetric positive definite matrix
%   Output:
%   X         - Samples matrix, dimension-by-number of samples
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maxNumCompThreads(2); 
% Size of random vector
xi = length(x0);  
vk = randn(xi,1); %0/randn(x1,1)*sqrt(sigma)/1e-3*gk
zk = randn(xi,1);
f  = zeros(N+1,1);
%% Initialisation
Ntau = length(stepsizek);     % Number of element of tauk
X = zeros(xi,N,Ntau);         % Samples
for s = 1:Ntau
    k               = 1;                % Step
    xk              = x0;               % Initial step
    stepsize        = stepsizek(s);     % Step-length
    % [ff,gk] = funh(xk);
    [ff,gk,Hess] = funh(xk);
    P=1./Hess;
    fprintf('Step: %d, misfit: %.4f\n', 0, ff);
    f(1)=ff;
    M_inv=1./M;
    M_sqrt=sqrt(M);

    tic;
    while k<=N


        %%
        % disp(['Step: ', num2str(k)]);
        % [xk1,vk1,zk1,fk1,gk1]=BACOCAB(xk,vk,zk,gama,zeta,stepsize,beta,M,P,funh,gk,xi,opt);
        [xk1,vk1,zk1,fk1,gk1,P]=OBCACBO(xk,vk,zk,gama,zeta,stepsize,beta,M_inv,M_sqrt,P,funh,gk,xi,opt);
        fprintf('Step: %d, misfit: %.4f\n', k, fk1);
        if isnan(fk1) || isinf(fk1)
            X(:,k,s)  = 0;
            break;
        end

        X(:,k,s)  = xk1;
        xk        = xk1;
        vk        = vk1;
        zk        = zk1;
        f(k+1)    = fk1;
        gk        = gk1;
        k       = k + 1;
    end
    toc;
end

end
%%
function [x1,v1,z1,fk1,gk1]=BACOCAB(x,v,z,gama,zeta,stepsize,beta,M,P,fh,gk,xi,opt)
    w=randn(xi,1);
    lmin=opt.down.*ones(size(x));
    lmax=opt.up.*ones(size(x));
    theta=exp(-stepsize*zeta);
    kapa =sqrt(1-theta^2);
    % kapa = sqrt((2 * (1 - theta)^2) / (stepsize*zeta));
    v1=v-.5*stepsize*P*gk;                                                               % B
    x1=x+.5*stepsize*P*(M\v1);                                                           % A
    [x1,v1,z]=boundReflection1(x1,v1,z,lmin,lmax,opt);
    v1=v1+.5*stepsize*gama*z;                                                            % C
    z1 = theta * z - (gama / zeta) * (1 - theta) .* v1 + sqrt(M * beta) * kapa  .* w;    % O
    v1=v1+.5*stepsize*gama*z1;                                                           % C
    x1=x1+.5*stepsize*P*(M\v1);                                                          % A
    [x1,v1,z1]=boundReflection1(x1,v1,z1,lmin,lmax,opt);
    [fk1,gk1]=fh(x1);
    v1=v1-.5*stepsize*P*gk;                                                              % B
    % [fk1,gk1]=fh(x1);
end

function [x1,v1,z1,fk1,gk1,P]=OBCACBO(x,v,z,gama,zeta,stepsize,beta,M_inv,M_sqrt,P,fh,gk,xi,opt)
    w1=randn(xi,1);
    w2=randn(xi,1);

    theta=exp(-0.5*stepsize*zeta);
    kapa =sqrt(1-theta^2);
    % kapa = sqrt((2 * (1 - theta)^2) / (stepsize * zeta));
    lmin=opt.down.*ones(size(x));
    lmax=opt.up.*ones(size(x));
    z1 = theta * z - (gama / zeta) * (1 - theta) .* v + sqrt(beta) * kapa * M_sqrt .* w1;    % O
    v1=v-.5*stepsize*P.*gk;                                                                  % B
    v1=v1+.5*stepsize*gama*z1;                                                               % C
    x1=x+stepsize*M_inv*P.*v1;                                                               % A
    % tic;[x1,v1,z1]=boundReflection(x1,v1,z1,lmin,lmax,opt);toc;
    [x1,v1,z1]=boundReflection1(x1,v1,z1,lmin,lmax);
    [fk1,gk1,Hess]=fh(x1);
    P=1./Hess;
    v1=v1+.5*stepsize*gama*z1;                                                               % C
    v1=v1-.5*stepsize*P.*gk1;                                                                % B
    z1 = theta * z1 - (gama / zeta) * (1 - theta) .* v1 + sqrt(beta) * kapa * M_sqrt .* w2;  % O
    % [fk1,gk1]=fh(x1);
end

function [m, v, z] = boundReflection1(m, v, z, lmin, lmax)

    % 循环直到所有点都在边界内
    while true
        % 计算超出上下界的部分
        upper_excess = m - lmax;
        lower_excess = lmin - m;
        % 处理上界反射
        upper_mask = upper_excess > 0;
        if any(upper_mask)
            m(upper_mask) = lmax(upper_mask) - (m(upper_mask) - lmax(upper_mask));
            % v(upper_mask) = -v(upper_mask)*0.1;
            % z(upper_mask) = -z(upper_mask)*0.1;
            v(upper_mask) = randn;
            z(upper_mask) = randn;
        end
        % 处理下界反射
        lower_mask = lower_excess > 0;
        if any(lower_mask)
            m(lower_mask) = lmin(lower_mask) + (lmin(lower_mask) - m(lower_mask));
            % v(lower_mask) = -v(lower_mask)*0.1;
            % z(lower_mask) = -z(lower_mask)*0.1;
            v(lower_mask) = randn;
            z(lower_mask) = randn;
        end
        % 检查是否所有点都在边界内
        if all(m >= lmin & m <= lmax)
            break;
        end
    end
end