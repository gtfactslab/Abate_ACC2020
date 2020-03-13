
% Paper:       "Lyapunov Differential Hierarchy and Polynomial Lyapunov 
%               Functions for Switched Linear Systems"
% Publisher:    2020 American Control Conference
% 
% Description: This script generates Figure 1 in the paper. Meta-Lyapunov
%            : functions of orders 2, 16, 20 and 26 are computed for a
%            : stable swithced linear system and then plotted.
%
% Code Author: Matthew Abate
% Date:        2/17/2020

clc; clf; clear all;


% -----------------
% System Parameters
% -----------------

% The system evolves subject to dx/dt = Ai x, for Ai \in \{A1, A2\}
global A1 A2 x0
A1 = [-0.5, 0.5; -0.5, -0.5];   % Switched mode 1
A2 = [-2.5, 2.5; -2.5,  1.5];   % Switched mode 2
x0 = [1; 0];                    % Initial condition (for levelsets)

c = [1, 5, 8, 13];      % C contains the levels of interest, i.e. the
                        % orders of the MLFs devided by 2

                        
% -----------------
% Setup Plot
% -----------------
figure(1); hold on; grid on;
set(gca, 'TickLabelInterpreter','latex','FontSize',18)
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
axis([-1.5, 1.5, -1.5, 1.5]);

load('ReachableSet.mat'); % Load reachable set (calculated seperately)
patch(States(1, 1:2:end), States(2, 1:2:end), ... % Plot the reachable set
            [240 230 180]/255, ... 
            'DisplayName', 'Reachable Set');

global color % These are the colors of the levelsets which are plotted
color = [44 123 182; ...
         171 217 233; ...
         253 174 97; ...
         215 25 28]/255;
 
 
% -----------------
% Find MLFs and Plot
% -----------------

syms x1 x2
X = [x1; x2]; % Generate symbolic state vector

for i = 1: size(c, 2)
    % Find levelset of 2c-th order MLF
    M = MetaLyap(c(i));

    % Plot
    if c(i) == 1
        plot(M(1, :), M(2, :), ...
            'Color', color(i, :), ...
            'LineWidth', 2, ...
            'DisplayName', [num2str(2*c(i), '%d') 'nd order']);
    else
        plot(M(1, :), M(2, :), ...
            'Color', color(i, :), ...
            'LineWidth', 2, ...
            'DisplayName', [num2str(2*c(i), '%d') 'th order']);
    end
    
    drawnow;
end


% Plot initial condition x0 = [1, 0];
scatter(x0(1), x0(2), 80, 'd', 'filled', ...
            'MarkerFaceColor', [0 .5 0], ...
            'DisplayName','Initial State');

legend('Interpreter','latex', 'Location', 'south', 'NumColumns', 2); % Generate Legend

grid on;
ax = gca;
ax.Layer = 'top';

% save figure to tikz
% matlab2tikz('Figure1.tikz', 'width', '7cm', ...
%                             'height', '5cm', ...
%                             'extraAxisOptions',{'axis on top'})

commandwindow;



% -----------------
% Functions
% -----------------

% MetaLyap takes in an integer c and searches for a metalyapunov function 
% of order 2c for the system.
% Input : Integer c
% Output: Level set of optimal 2c-th order MLF
function M = MetaLyap(c)
    global A1 A2 x0
    
    % Calculate Wc from Proposition 2 (See Paper) 
    W = eye(2);
    for i = 2:c
        W = [[W,  zeros(2^(i-1), 1)]; [zeros(2^(i-1), 1), W]];
    end
    
    % Comput Bc from Wc and Ac
    Ac1 = sparse(A1);
    Ac2 = sparse(A2);
    for i = 1:c-1
        % Compute Ac - sparse matrixes are used for computation speed
        Ac1 = kron(speye(2), Ac1) + kron(A1, speye(2^i));
        Ac2 = kron(speye(2), Ac2) + kron(A2, speye(2^i));
    end
    % Assign Bc
    Bc1 = pinv(W)*Ac1*W;
    Bc2 = pinv(W)*Ac2*W;

    %Search for common Lyapunov function using Algorithm 1
    cvx_begin sdp
        %Optimize over P
        variable P(c+1,c+1) semidefinite

        %Constraints
        0 >= Bc1'*P + P*Bc1;
        0 >= Bc2'*P + P*Bc2;
        P >= eye(c+1);

        minimize(P(1, 1));
    cvx_end
    cvx_cputime

    if isequal(cvx_status, 'Failed') || isequal(cvx_status, 'Infeasible')
        error('Solution not found');
    end
    
    
    % We next approximate the level set of the MLF which goes through x0
    
    x1 = -2: .005 : 2; x2 = -2: .005 : 2; % Discritize Domain
    [X1r, X2r] = meshgrid(x1, x2); 

    for i = 1:size(X1r, 1)
        for j = 1:size(X1r, 2)

            x = [];
            for g = 0:c
                x(g+1, 1) = X1r(i, j)^(c - g)*X2r(i, j)^(g);
            end

            Zr(i, j) = x.'*P*x;
        end
    end
    x0m = [];
    for g = 0:c
        x0m(g+1, 1) = x0(1)^(c - g)*x0(2)^(g);
    end
    l = x0m.'*P*x0m;
    [M, c] = contour(X1r, X2r, Zr, [l, l], 'HandleVisibility', 'off');
    delete(c)
    M = M(:, 2:end); % return level set
end
