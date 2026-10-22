poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

filePath = "C:\Users\sahil\OneDrive\Desktop\Old_DOE.xlsx"; 
try
    data = readmatrix(filePath);
    X = data(:, 1:5); 
    Y_hardness = data(:, 6);
    Y_tensile = data(:, 7);
catch
    warning('Dataset not found at specified path. Generating placeholder for structure testing.');
    X = rand(45, 5) * 100;
    Y_hardness = 100 - 0.3*X(:,1) + 0.4*X(:,2) + randn(45,1)*5;
    Y_tensile = 50 + 0.5*X(:,1) - 0.2*X(:,2) + randn(45,1)*5;
end

Y = [Y_hardness, Y_tensile];
[N_runs, num_vars] = size(X);
fprintf('Successfully loaded dataset: %d runs and %d input factors.\n', N_runs, num_vars);

disp('Training surrogate models mapping inputs to responses...');
mdl_hardness = fitlm(X, Y(:,1), 'quadratic');
mdl_tensile = fitlm(X, Y(:,2), 'quadratic');

% Objective Function: Minimize Hardness, Maximize Tensile (via negation)
ObjFun = @(x) [-predict(mdl_hardness, x), -predict(mdl_tensile, x)];

disp('Initializing Hybrid MO-ABC-HHO Algorithm in Parallel...');
SearchAgents_no = 50; 
Max_iter = 100;     
dim = num_vars;       
lb = min(X);          
ub = max(X); 

% Hybrid Specific Parameter
limit = 50; % ABC Scout Bee abandonment limit

num_runs = 10;
best_params_all = zeros(num_runs, dim);
best_responses_all = zeros(num_runs, 2);
all_Pareto_hardness = [];
all_Pareto_tensile = [];

disp(['Running MO-ABC-HHO Iterations ', num2str(num_runs), ' times...']);

% --- 2. MULTI-RUN OPTIMIZATION LOOP ---
for r = 1:num_runs
    fprintf('Executing Run %d of %d...\n', r, num_runs);
    
    % Initialize Positions and Trials
    Positions = repmat(lb, SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* repmat((ub - lb), SearchAgents_no, 1);
    Fitness = zeros(SearchAgents_no, 2);
    trial = zeros(SearchAgents_no, 1);
    
    % Initial Evaluation
    parfor i = 1:SearchAgents_no
        pos = max(min(Positions(i,:), ub), lb);
        Fitness(i,:) = ObjFun(pos);
    end
    
    Archive_X = Positions;
    Archive_F = Fitness;
    
    % Initial Non-dominated Sorting for the Archive
    [front, ~] = checkDomination(Archive_F);
    Archive_X = Archive_X(front, :);
    Archive_F = Archive_F(front, :);
    
    Convergence_curve = zeros(Max_iter, 2);
    
    for iter = 1:Max_iter
        
        new_Positions = zeros(SearchAgents_no, dim);
        X_mean = mean(Positions, 1); 
        
        for i = 1:SearchAgents_no
            % Select a target (rabbit) randomly from the Pareto Archive
            rabbit_idx = randi(size(Archive_X, 1));
            X_rabbit = Archive_X(rabbit_idx, :);
            
            % Calculate Escaping Energy (HHO Core)
            E0 = 2 * rand() - 1; 
            E = 2 * E0 * (1 - iter / Max_iter); 
            J = 2 * (1 - rand()); 
            
            % HYBRID SWITCHING MECHANISM
            if abs(E) >= 1 
                % --- ABC EXPLORATION PHASE (Employed Bee) ---
                k = randi(SearchAgents_no);
                while k == i, k = randi(SearchAgents_no); end
                
                j = randi(dim); % Random dimension to perturb
                phi = (rand() - 0.5) * 2; 
                
                new_Positions(i,:) = Positions(i,:);
                new_Positions(i,j) = Positions(i,j) + phi * (Positions(i,j) - Positions(k,j));
                
            else 
                % --- HHO EXPLOITATION PHASE (Besiege Tactics) ---
                prob_r = rand(); 
                
                if prob_r >= 0.5 && abs(E) >= 0.5 % Soft besiege
                    new_Positions(i,:) = (X_rabbit - Positions(i,:)) - E * abs(J * X_rabbit - Positions(i,:));
                    
                elseif prob_r >= 0.5 && abs(E) < 0.5 % Hard besiege
                    new_Positions(i,:) = X_rabbit - E * abs(X_rabbit - Positions(i,:));
                    
                elseif prob_r < 0.5 && abs(E) >= 0.5 % Soft besiege with rapid dives
                    Y_pos = X_rabbit - E * abs(J * X_rabbit - Positions(i,:));
                    Z_pos = Y_pos + rand(1, dim) .* levy_flight(dim);
                    if rand() < 0.5, new_Positions(i,:) = Y_pos; else, new_Positions(i,:) = Z_pos; end
                    
                elseif prob_r < 0.5 && abs(E) < 0.5 % Hard besiege with rapid dives
                    Y_pos = X_rabbit - E * abs(J * X_rabbit - X_mean);
                    Z_pos = Y_pos + rand(1, dim) .* levy_flight(dim);
                    if rand() < 0.5, new_Positions(i,:) = Y_pos; else, new_Positions(i,:) = Z_pos; end
                end
            end
        end
        
        % Safeguard against NaN/Inf and Enforce Boundaries
        new_Positions(isnan(new_Positions) | isinf(new_Positions)) = lb(1);
        new_Positions = max(min(new_Positions, ub), lb);
        
        % Evaluate New Positions
        new_Fitness = zeros(SearchAgents_no, 2);
        parfor i = 1:SearchAgents_no
            new_Fitness(i,:) = ObjFun(new_Positions(i,:));
        end
        
        % Dominance-based Selection and Trial Update
        for i = 1:SearchAgents_no
            dom1 = (new_Fitness(i,1) <= Fitness(i,1) && new_Fitness(i,2) <= Fitness(i,2)) && ...
                   (new_Fitness(i,1) < Fitness(i,1) || new_Fitness(i,2) < Fitness(i,2));
            dom2 = (Fitness(i,1) <= new_Fitness(i,1) && Fitness(i,2) <= new_Fitness(i,2)) && ...
                   (Fitness(i,1) < new_Fitness(i,1) || Fitness(i,2) < new_Fitness(i,2));
                   
            if dom1 
                Positions(i,:) = new_Positions(i,:);
                Fitness(i,:) = new_Fitness(i,:);
                trial(i) = 0; % Reset trial counter
            elseif ~dom2 && rand() < 0.5 
                Positions(i,:) = new_Positions(i,:);
                Fitness(i,:) = new_Fitness(i,:);
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1; % Increment trial counter
            end
        end
        
        % --- ABC SCOUT BEE PHASE ---
        % Replace solutions trapped in local optima
        scout_idx = find(trial > limit);
        if ~isempty(scout_idx)
            scout_Positions = zeros(length(scout_idx), dim);
            for idx = 1:length(scout_idx)
                scout_Positions(idx,:) = lb + rand(1, dim) .* (ub - lb);
            end
            
            scout_Fitness = zeros(length(scout_idx), 2);
            parfor idx = 1:length(scout_idx)
                scout_Fitness(idx,:) = ObjFun(scout_Positions(idx,:));
            end
            
            for idx = 1:length(scout_idx)
                orig_i = scout_idx(idx);
                Positions(orig_i,:) = scout_Positions(idx,:);
                Fitness(orig_i,:) = scout_Fitness(idx,:);
                trial(orig_i) = 0;
            end
        end
        
        % --- EXTERNAL ARCHIVE UPDATE ---
        combined_X = [Archive_X; Positions];
        combined_F = [Archive_F; Fitness];
        
        [front, ~] = checkDomination(combined_F);
        Archive_X = combined_X(front, :);
        Archive_F = combined_F(front, :);
      
        % Archive Truncation
        Max_Archive = 100;
        if size(Archive_X, 1) > Max_Archive
            keep_idx = randperm(size(Archive_X, 1), Max_Archive);
            Archive_X = Archive_X(keep_idx, :);
            Archive_F = Archive_F(keep_idx, :);
        end
        
        Convergence_curve(iter, 1) = mean(Archive_F(:,1));
        Convergence_curve(iter, 2) = mean(Archive_F(:,2));
    end
    
    % Store True Pareto Outputs
    Pareto_hardness = Archive_F(:,1); 
    Pareto_tensile = Archive_F(:,2);
    
    all_Pareto_hardness = [all_Pareto_hardness; Pareto_hardness];
    all_Pareto_tensile = [all_Pareto_tensile; Pareto_tensile];
    
    % Calculate Ideal Point Distances
    range_H = max(Pareto_hardness) - min(Pareto_hardness);
    range_T = max(Pareto_tensile) - min(Pareto_tensile);
    if range_H == 0 || isnan(range_H), range_H = 1; end 
    if range_T == 0 || isnan(range_T), range_T = 1; end
    
    Norm_H = (Pareto_hardness - min(Pareto_hardness)) ./ range_H;
    Norm_T = (Pareto_tensile - min(Pareto_tensile)) ./ range_T;
    
    Dist_to_Ideal = sqrt((0 - Norm_H).^2 + (0 - Norm_T).^2);
    [~, best_idx] = min(Dist_to_Ideal);
   
    if isempty(best_idx) || isnan(best_idx)
        best_idx = 1; 
    end
    
    best_params_all(r, :) = Archive_X(best_idx, :);
    best_responses_all(r, :) = [Pareto_hardness(best_idx), Pareto_tensile(best_idx)];
end

avg_optimal_params = mean(best_params_all, 1);
avg_optimal_responses = mean(best_responses_all, 1);

% --- 3. RESULTS CONSOLE OUTPUT ---
disp('----------------------------------------------------');
disp('Best Compromise Solution from Each of the 10 Runs (MO-ABC-HHO):');
disp('Run |   NT     PS     RD     BT     FR    |      Hardness   Tensile');
disp('----------------------------------------------------');
for r = 1:num_runs
    fprintf('%3d | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
        r, best_params_all(r,:), abs(best_responses_all(r,1)), abs(best_responses_all(r,2)));
end
disp('----------------------------------------------------');
disp('AVERAGE OPTIMAL VALUES ACROSS 10 RUNS:');
fprintf('AVG | %8.3f %9.3f %9.3f %9.3f %9.3f | %8.3f %9.3f\n', ...
    avg_optimal_params, abs(avg_optimal_responses(1)), abs(avg_optimal_responses(2)));
disp('----------------------------------------------------');

% --- 4. VISUALIZATIONS ---
figure('Name', 'Pareto Optimal Front (Combined)', 'Color', 'w');
plot(abs(all_Pareto_hardness), abs(all_Pareto_tensile), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b'); hold on;
plot(abs(best_responses_all(:,1)), abs(best_responses_all(:,2)), 'g*', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(abs(avg_optimal_responses(1)), abs(avg_optimal_responses(2)), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
xlabel('Hardness (Minimize)');
ylabel('Tensile Strength (Maximize)');
title('Pareto Optimal Fronts across 10 Runs (MO-ABC-HHO)');
legend('All Pareto Solutions', 'Best per Run', 'Average Optimal Value', 'Location', 'Best');
grid on;

figure('Name', 'Convergence Plot (Last Run)', 'Color', 'w');
yyaxis left;
plot(1:Max_iter, abs(Convergence_curve(:,1)), 'b-', 'LineWidth', 2);
ylabel('Avg Hardness (Archive)');
yyaxis right;
plot(1:Max_iter, abs(Convergence_curve(:,2)), 'r-', 'LineWidth', 2);
ylabel('Avg Tensile Strength (Archive)');
xlabel('Iteration');
title('Algorithm Convergence over Iterations (Final Run)');
grid on;

x1_vec = linspace(lb(1), ub(1), 30);
x2_vec = linspace(lb(2), ub(2), 30);
[X1_mat, X2_mat] = meshgrid(x1_vec, x2_vec);
mean_x3 = mean(X(:,3)); mean_x4 = mean(X(:,4)); mean_x5 = mean(X(:,5));
Z_hardness = zeros(size(X1_mat));
Z_tensile = zeros(size(X1_mat));

for i = 1:numel(X1_mat)
    pt = [X1_mat(i), X2_mat(i), mean_x3, mean_x4, mean_x5];
    Z_hardness(i) = predict(mdl_hardness, pt);
    Z_tensile(i) = predict(mdl_tensile, pt);
end

figure('Name', '3D Surface Plots', 'Position', [100 100 1000 400], 'Color', 'w');
subplot(1,2,1);
surf(X1_mat, X2_mat, Z_hardness, 'EdgeColor', 'none');
colormap(gca, parula); colorbar;
xlabel('Factor 1'); ylabel('Factor 2'); zlabel('Hardness');
title('Response Surface: Hardness');
subplot(1,2,2);
surf(X1_mat, X2_mat, Z_tensile, 'EdgeColor', 'none');
colormap(gca, hot); colorbar;
xlabel('Factor 1'); ylabel('Factor 2'); zlabel('Tensile Strength');
title('Response Surface: Tensile Strength');

% --- HELPER FUNCTIONS ---
function [front_idx, domination_count] = checkDomination(F)
    pop_size = size(F, 1);
    domination_count = zeros(pop_size, 1);
    for i = 1:pop_size
        for j = 1:pop_size
            if i ~= j
                if (F(j,1) <= F(i,1) && F(j,2) <= F(i,2)) && ...
                   (F(j,1) < F(i,1) || F(j,2) < F(i,2))
                    domination_count(i) = domination_count(i) + 1;
                end
            end
        end
    end
    front_idx = find(domination_count == 0);
end

function LF = levy_flight(dim)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
            (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    step = u ./ (abs(v).^(1 / beta) + eps); 
    LF = 0.01 * step;
end