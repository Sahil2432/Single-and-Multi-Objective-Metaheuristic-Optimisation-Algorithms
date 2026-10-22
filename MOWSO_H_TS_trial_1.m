poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

filePath = "C:\Users\sahil\OneDrive\Desktop\NEW_DOE_dataset.xlsx"; 
try
    data = readmatrix(filePath);
    X = data(:, 1:5);          
    Y_hardness = data(:, 6);     
    Y_tensile = data(:, 7);    
catch
    warning('Dataset not found. Please ensure the file path is correct.');
end

Y = [Y_hardness, Y_tensile];
[N_runs, num_vars] = size(X);

disp('Training surrogate models mapping inputs to responses...');
mdl_hardness = fitlm(X, Y(:,1), 'quadratic');
mdl_tensile = fitlm(X, Y(:,2), 'quadratic');

% ObjFun: Column 1 is Hardness (Minimize), Column 2 is -Tensile (Minimize to Maximize)
ObjFun = @(x) [predict(mdl_hardness, x), -predict(mdl_tensile, x)];

disp('Initializing Multi-Objective White Shark Optimizer (MOWSO)...');
lb = min(X); 
ub = max(X); 

% MOWSO Parameters
SearchAgents_no = 50;
Max_iter = 100;
Archive_size = 50;
num_MOWSO_runs = 10;

best_params_all = zeros(num_MOWSO_runs, num_vars);
best_responses_all = zeros(num_MOWSO_runs, 2);
all_Pareto_hardness = [];
all_Pareto_tensile = [];

disp(['Running MOWSO Iterations ', num2str(num_MOWSO_runs), ' times...']);

for r = 1:num_MOWSO_runs
    fprintf('Executing Run %d of %d...\n', r, num_MOWSO_runs);
    
    [Archive_X, Archive_F, hist_F1, hist_F2] = run_MOWSO(ObjFun, num_vars, lb, ub, SearchAgents_no, Max_iter, Archive_size);
    
    % Extract true positive values: Col 1 is Hardness, Col 2 is -Tensile
    Pareto_hardness = Archive_F(:,1);
    Pareto_tensile = -Archive_F(:,2); 
    
    % Store all Pareto fronts for the combined plot
    all_Pareto_hardness = [all_Pareto_hardness; Pareto_hardness];
    all_Pareto_tensile = [all_Pareto_tensile; Pareto_tensile];
    
    % Normalize to find the best compromise solution for THIS run
    range_H = max(Pareto_hardness) - min(Pareto_hardness);
    range_T = max(Pareto_tensile) - min(Pareto_tensile);
    if range_H == 0, range_H = 1; end 
    if range_T == 0, range_T = 1; end
    
    Norm_H = (Pareto_hardness - min(Pareto_hardness)) ./ range_H;
    Norm_T = (Pareto_tensile - min(Pareto_tensile)) ./ range_T;
    
    % Distance to ideal metric: Target 0 for Norm_H (Min), Target 1 for Norm_T (Max)
    Dist_to_Ideal = sqrt((0 - Norm_H).^2 + (1 - Norm_T).^2);
    [~, best_idx] = min(Dist_to_Ideal);
    
    % Store the single best compromise solution
    best_params_all(r, :) = Archive_X(best_idx, :);
    best_responses_all(r, :) = [Pareto_hardness(best_idx), Pareto_tensile(best_idx)];
end

% Calculate the average across the 10 runs
avg_optimal_params = mean(best_params_all, 1);
avg_optimal_responses = mean(best_responses_all, 1);

disp('----------------------------------------------------');
disp('Best Compromise Solution from Each of the 10 Runs (MOWSO):');
disp('Run |   NT     PS     RD     BT      FR    | Hardness Tensile');
disp('----------------------------------------------------');
for r = 1:num_MOWSO_runs
    % Removed abs() - variables are explicitly handled as positive now
    fprintf('%3d | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
        r, best_params_all(r,:), best_responses_all(r,1), best_responses_all(r,2));
end
disp('----------------------------------------------------');
disp('AVERAGE OPTIMAL VALUES ACROSS 10 RUNS:');
fprintf('AVG | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
    avg_optimal_params, avg_optimal_responses(1), avg_optimal_responses(2));
disp('----------------------------------------------------');

% PARETO OPTIMAL FRONT PLOT
figure('Name', 'MOWSO Pareto Optimal Front (Combined)', 'Color', 'w');
plot(all_Pareto_hardness, all_Pareto_tensile, 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b'); hold on;
plot(best_responses_all(:,1), best_responses_all(:,2), 'g*', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(avg_optimal_responses(1), avg_optimal_responses(2), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlabel('Hardness (Minimize)'); ylabel('Tensile Strength (Maximize)');
title('Pareto Optimal Fronts across 10 Runs (MOWSO)');
legend('All Pareto Solutions', 'Best per Run', 'Average Optimal Value', 'Location', 'Best'); grid on;

% CONVERGENCE PLOT (Using the history from the final run)
figure('Name', 'MOWSO Convergence Plot (Last Run)', 'Color', 'w');
yyaxis left;
plot(1:length(hist_F1), hist_F1, 'b-', 'LineWidth', 2);
ylabel('Avg Hardness (Archive Front)');

yyaxis right;
plot(1:length(hist_F2), hist_F2, 'r-', 'LineWidth', 2);
ylabel('Avg Tensile Strength (Archive Front)');

xlabel('Iteration');
title('MOWSO Convergence over Iterations (Final Run)');
grid on;

% 3D SURFACE PLOTS
x1_vec = linspace(lb(1), ub(1), 30);
x2_vec = linspace(lb(2), ub(2), 30);
[X1_mat, X2_mat] = meshgrid(x1_vec, x2_vec);
mean_x3 = mean(X(:,3)); mean_x4 = mean(X(:,4)); mean_x5 = mean(X(:,5));
Z_hardness = zeros(size(X1_mat)); Z_tensile = zeros(size(X1_mat));
for i = 1:numel(X1_mat)
    pt = [X1_mat(i), X2_mat(i), mean_x3, mean_x4, mean_x5];
    Z_hardness(i) = predict(mdl_hardness, pt);
    Z_tensile(i) = predict(mdl_tensile, pt);
end

figure('Name', '3D Surface Plots', 'Position', [100 100 1000 400], 'Color', 'w');
subplot(1,2,1);
surf(X1_mat, X2_mat, Z_hardness, 'EdgeColor', 'none'); colormap(gca, parula); colorbar;
xlabel('Nozzle temperature'); ylabel('Print Speed'); zlabel('Hardness');
title('Response Surface: Hardness');
subplot(1,2,2);
surf(X1_mat, X2_mat, Z_tensile, 'EdgeColor', 'none'); colormap(gca, hot); colorbar;
xlabel('Nozzle Temperature'); ylabel('Print Speed'); zlabel('Tensile Strength');
title('Response Surface: Tensile Strength');

% =========================================================================
% HELPER FUNCTIONS FOR MOWSO
% =========================================================================

function [Archive_X, Archive_F, hist_F1, hist_F2] = run_MOWSO(ObjFun, nVar, lb, ub, N, MaxIt, ArchiveSize)
    % Initialize Population (Sharks)
    X = repmat(lb, N, 1) + repmat((ub - lb), N, 1) .* rand(N, nVar);
    V = zeros(N, nVar); % Velocities
    F = zeros(N, 2);
    for i=1:N
        F(i,:) = ObjFun(X(i,:));
    end
    [Archive_X, Archive_F] = UpdateArchive(X, F, [], [], ArchiveSize);
    
    hist_F1 = zeros(MaxIt, 1);
    hist_F2 = zeros(MaxIt, 1);
    
    % WSO specific baseline parameters
    p_min = 0.5; p_max = 1.5;
    a1 = 35;
    
    for it = 1:MaxIt
        % Select Global Best (Prey) from Archive using Crowding Distance
        cd = calcCrowdingDistance(Archive_F);
        [~, sortIdx] = sort(cd, 'descend');
        if length(sortIdx) >= 1
            G_best = Archive_X(sortIdx(1), :);
        else
            G_best = X(1,:);
        end
        
        for i = 1:N
            % Update parameters for WSO exploration/exploitation
            p1 = p_max + (p_max - p_min) * exp(-(4 * it / MaxIt)^2);
            p2 = p_min + (p_max - p_min) * exp(-(4 * it / MaxIt)^2);
            mu = 2 * rand() - 1; 
            
            for j = 1:nVar
                rand1 = rand(); rand2 = rand(); rand3 = rand();
                
                % White Shark Velocity Update Equation
                rand_arc_idx = randi(size(Archive_X,1));
                V(i,j) = mu * V(i,j) + p1 * rand1 * (G_best(j) - X(i,j)) + p2 * rand2 * (Archive_X(rand_arc_idx, j) - X(i,j));
                
                % White Shark Position Update Equation
                if rand() < 0.5
                    X(i,j) = X(i,j) + V(i,j) * rand3;
                else
                    X(i,j) = G_best(j) + V(i,j) * rand3;
                end
            end
        end
        
        % Boundary checking
        X = max(X, lb);
        X = min(X, ub);
        
        for i = 1:N
            F(i,:) = ObjFun(X(i,:));
        end
        
        [Archive_X, Archive_F] = UpdateArchive(X, F, Archive_X, Archive_F, ArchiveSize);
        
        % Store the average of the *Archive* (Pareto Front) to properly show convergence
        hist_F1(it) = mean(Archive_F(:,1));  % Average Hardness 
        hist_F2(it) = mean(-Archive_F(:,2)); % Average True Tensile (flipping sign)
    end
end

function [NewArc_X, NewArc_F] = UpdateArchive(X, F, Arc_X, Arc_F, MaxSize)
    All_X = [X; Arc_X];
    All_F = [F; Arc_F];
    
    % Fast Non-dominated sort 
    N = size(All_F, 1);
    domination_count = zeros(N, 1);
    for i = 1:N
        for j = 1:N
            if i~=j
                if all(All_F(j,:) <= All_F(i,:)) && any(All_F(j,:) < All_F(i,:))
                    domination_count(i) = domination_count(i) + 1;
                end
            end
        end
    end
    
    nd_idx = find(domination_count == 0);
    NewArc_X = All_X(nd_idx, :);
    NewArc_F = All_F(nd_idx, :);
    
    % Ensure unique solutions
    [NewArc_F, unique_idx] = unique(NewArc_F, 'rows');
    NewArc_X = NewArc_X(unique_idx, :);
    
    % Truncate using Crowding Distance if exceeding MaxSize
    if size(NewArc_X, 1) > MaxSize
        cd = calcCrowdingDistance(NewArc_F);
        [~, sortIdx] = sort(cd, 'descend');
        NewArc_X = NewArc_X(sortIdx(1:MaxSize), :);
        NewArc_F = NewArc_F(sortIdx(1:MaxSize), :);
    end
end

function cd = calcCrowdingDistance(F)
    N = size(F,1);
    M = size(F,2);
    cd = zeros(N,1);
    for m = 1:M
        [sorted_F, idx] = sort(F(:,m));
        cd(idx(1)) = inf;
        cd(idx(end)) = inf;
        f_min = sorted_F(1);
        f_max = sorted_F(end);
        if f_max > f_min
            for i = 2:N-1
                cd(idx(i)) = cd(idx(i)) + (sorted_F(i+1) - sorted_F(i-1)) / (f_max - f_min);
            end
        end
    end
end