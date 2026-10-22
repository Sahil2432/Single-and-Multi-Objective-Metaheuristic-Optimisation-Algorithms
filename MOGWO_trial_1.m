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
end

Y = [Y_hardness, Y_tensile];
[N_runs, num_vars] = size(X);

disp('Training surrogate models mapping inputs to responses...');
mdl_hardness = fitlm(X, Y(:,1), 'quadratic');
mdl_tensile = fitlm(X, Y(:,2), 'quadratic');
ObjFun = @(x) [predict(mdl_hardness, x), -predict(mdl_tensile, x)];

disp('Initializing Multi-Objective Grey Wolf Optimizer (MOGWO)...');
lb = min(X); 
ub = max(X); 

% MOGWO Parameters
SearchAgents_no = 50;
Max_iter = 100;
Archive_size = 50;
num_MOGWO_runs = 10;

best_params_all = zeros(num_MOGWO_runs, num_vars);
best_responses_all = zeros(num_MOGWO_runs, 2);
all_Pareto_hardness = [];
all_Pareto_tensile = [];

disp(['Running MOGWO Iterations ', num2str(num_MOGWO_runs), ' times...']);

for r = 1:num_MOGWO_runs
    fprintf('Executing Run %d of %d...\n', r, num_MOGWO_runs);
    
    [Archive_X, Archive_F, hist_F1, hist_F2] = run_MOGWO(ObjFun, num_vars, lb, ub, SearchAgents_no, Max_iter, Archive_size);
    
    Pareto_hardness = -Archive_F(:,1);
    Pareto_tensile = Archive_F(:,2);
    
    % Store all Pareto fronts for the combined plot
    all_Pareto_hardness = [all_Pareto_hardness; Pareto_hardness];
    all_Pareto_tensile = [all_Pareto_tensile; Pareto_tensile];
    
    % Normalize to find the best compromise solution for THIS run
    range_H = max(Pareto_hardness) - min(Pareto_hardness);
    range_T = max(Pareto_tensile) - min(Pareto_tensile);
    if range_H == 0, range_H = 1; end % Prevent division by zero
    if range_T == 0, range_T = 1; end
    
    Norm_H = (Pareto_hardness - min(Pareto_hardness)) ./ range_H;
    Norm_T = (Pareto_tensile - min(Pareto_tensile)) ./ range_T;
    
    % Distance to ideal metric
    Dist_to_Ideal = sqrt((1 - Norm_H).^2 + (0 - Norm_T).^2);
    [~, best_idx] = min(Dist_to_Ideal);
    
    % Store the single best compromise solution
    best_params_all(r, :) = Archive_X(best_idx, :);
    best_responses_all(r, :) = [Pareto_hardness(best_idx), Pareto_tensile(best_idx)];
end

% Calculate the average across the 10 runs
avg_optimal_params = mean(best_params_all, 1);
avg_optimal_responses = mean(best_responses_all, 1);

disp('----------------------------------------------------');
disp('Best Compromise Solution from Each of the 10 Runs (MOGWO):');
disp('Run |   Var1      Var2      Var3      Var4      Var5    | Hardness Tensile');
disp('----------------------------------------------------');
for r = 1:num_MOGWO_runs
    fprintf('%3d | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
        r, best_params_all(r,:), abs(best_responses_all(r,1)), abs(best_responses_all(r,2)));
end
disp('----------------------------------------------------');
disp('AVERAGE OPTIMAL VALUES ACROSS 10 RUNS:');
fprintf('AVG | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
    avg_optimal_params, abs(avg_optimal_responses(1)), abs(avg_optimal_responses(2)));
disp('----------------------------------------------------');

% PARETO OPTIMAL FRONT PLOT
figure('Name', 'MOGWO Pareto Optimal Front (Combined)', 'Color', 'w');
plot(abs(all_Pareto_hardness), abs(all_Pareto_tensile), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b'); hold on;
plot(abs(best_responses_all(:,1)), abs(best_responses_all(:,2)), 'g*', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(abs(avg_optimal_responses(1)), abs(avg_optimal_responses(2)), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlabel('Hardness (Minimize)'); ylabel('Tensile Strength (Maximize)');
title('Pareto Optimal Fronts across 10 Runs (MOGWO)');
legend('All Pareto Solutions', 'Best per Run', 'Average Optimal Value', 'Location', 'Best'); grid on;

% CONVERGENCE PLOT (Using the history from the final run)
figure('Name', 'MOGWO Convergence Plot', 'Color', 'w');
yyaxis left;
plot(1:length(hist_F1), abs(hist_F1), 'b-', 'LineWidth', 2);
ylabel('Avg Hardness (Population)');

yyaxis right;
plot(1:length(hist_F2), abs(hist_F2), 'r-', 'LineWidth', 2);
ylabel('Avg Tensile Strength (Population)');

xlabel('Iteration');
title('MOGWO Convergence over Iterations');
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

% HELPER FUNCTIONS FOR MOGWO

function [Archive_X, Archive_F, hist_F1, hist_F2] = run_MOGWO(ObjFun, nVar, lb, ub, N, MaxIt, ArchiveSize)
    % Initialize Population
    X = repmat(lb, N, 1) + repmat((ub - lb), N, 1) .* rand(N, nVar);
    F = zeros(N, 2);
    for i=1:N
        F(i,:) = ObjFun(X(i,:));
    end
    [Archive_X, Archive_F] = UpdateArchive(X, F, [], [], ArchiveSize);
    
    hist_F1 = zeros(MaxIt, 1);
    hist_F2 = zeros(MaxIt, 1);
    
    for it = 1:MaxIt
        a = 2 - it * (2 / MaxIt); % Linearly decreases from 2 to 0
        
        % Select Alpha, Beta, Delta from Archive using Crowding Distance
        cd = calcCrowdingDistance(Archive_F);
        [~, sortIdx] = sort(cd, 'descend');
        
        if length(sortIdx) >= 1, Alpha_Pos = Archive_X(sortIdx(1), :); else, Alpha_Pos = X(1,:); end
        if length(sortIdx) >= 2, Beta_Pos = Archive_X(sortIdx(2), :); else, Beta_Pos = Alpha_Pos; end
        if length(sortIdx) >= 3, Delta_Pos = Archive_X(sortIdx(3), :); else, Delta_Pos = Beta_Pos; end
        
        for i = 1:N
            for j = 1:nVar
                r1 = rand(); r2 = rand();
                A1 = 2*a*r1 - a; C1 = 2*r2;
                D_alpha = abs(C1*Alpha_Pos(j) - X(i,j));
                X1 = Alpha_Pos(j) - A1*D_alpha;
                
                r1 = rand(); r2 = rand();
                A2 = 2*a*r1 - a; C2 = 2*r2;
                D_beta = abs(C2*Beta_Pos(j) - X(i,j));
                X2 = Beta_Pos(j) - A2*D_beta;
                
                r1 = rand(); r2 = rand();
                A3 = 2*a*r1 - a; C3 = 2*r2;
                D_delta = abs(C3*Delta_Pos(j) - X(i,j));
                X3 = Delta_Pos(j) - A3*D_delta;
                
                X(i,j) = (X1 + X2 + X3) / 3;
            end
        end
        
        % Boundary checking
        X = max(X, lb);
        X = min(X, ub);
        
        for i = 1:N
            F(i,:) = ObjFun(X(i,:));
        end
        
        [Archive_X, Archive_F] = UpdateArchive(X, F, Archive_X, Archive_F, ArchiveSize);
        
        hist_F1(it) = mean(F(:,1));
        hist_F2(it) = mean(F(:,2));
    end
end

function [NewArc_X, NewArc_F] = UpdateArchive(X, F, Arc_X, Arc_F, MaxSize)
    All_X = [X; Arc_X];
    All_F = [F; Arc_F];
        
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