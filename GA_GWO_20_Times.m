% =========================================================================
% Hybrid GA-GWO Optimization (Automated for 20 Independent Runs)
% =========================================================================

% 1. Import and Prepare the Dataset
filename = 'Tribological_dataset.xlsx'; 
data = readmatrix(filename);

% Extract Inputs (Columns 1-5) and Outputs (Columns 6-7)
inputs = data(:, 1:5); 
outputs = data(:, 6:7);
target_output = outputs(:, 1); % Optimizing for Frictional force

% 2. Master Setup
num_runs = 20;              % Number of independent runs for statistical validation
SearchAgents_no = 30;       % Population size
Max_iter = 100;             % Maximum number of iterations per run
dim = 5;                    % Number of input parameters
lb = [190, 20, 2, 40, 90];  % Lower bounds
ub = [230, 60, 8, 70, 110]; % Upper bounds
crossover_rate = 0.7;       % GA Crossover Probability
mutation_rate = 0.1;        % GA Mutation Probability

% Storage for statistical analysis
all_runs_best_costs = zeros(num_runs, 1);
all_runs_best_params = zeros(num_runs, dim);
all_convergence_curves = zeros(num_runs, Max_iter);

% Start parallel pool for faster execution across all runs
if isempty(gcp('nocreate'))
    parpool('local'); 
end

fprintf('Starting %d independent runs of GA-GWO...\n\n', num_runs);

% =========================================================================
% 3. Main Multi-Run Loop
% =========================================================================
for current_run = 1:num_runs
    fprintf('Executing Run %d of %d...\n', current_run, num_runs);
    
    % Reset Alpha, Beta, and Delta wolves for the new run
    Alpha_pos = zeros(1, dim);  Alpha_score = inf; 
    Beta_pos = zeros(1, dim);   Beta_score = inf;
    Delta_pos = zeros(1, dim);  Delta_score = inf;

    % Re-initialize Population randomly
    Positions = initialization(SearchAgents_no, dim, ub, lb);
    Convergence_curve = zeros(1, Max_iter);

    % Optimization Loop for the current run
    for l = 1:Max_iter
        
        % --- FITNESS EVALUATION (Parallelized) ---
        fitness = zeros(1, SearchAgents_no);
        parfor i = 1:size(Positions, 1)
            % Boundary checking
            Flag4ub = Positions(i,:) > ub;
            Flag4lb = Positions(i,:) < lb;
            Positions(i,:) = (Positions(i,:).*(~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;               
            
            % Calculate objective function
            fitness(i) = your_objective_function(Positions(i,:), inputs, target_output);
        end
        
        % --- UPDATE ALPHA, BETA, DELTA ---
        for i = 1:SearchAgents_no
            if fitness(i) < Alpha_score 
                Alpha_score = fitness(i); 
                Alpha_pos = Positions(i,:);
            elseif fitness(i) > Alpha_score && fitness(i) < Beta_score 
                Beta_score = fitness(i); 
                Beta_pos = Positions(i,:);
            elseif fitness(i) > Alpha_score && fitness(i) > Beta_score && fitness(i) < Delta_score 
                Delta_score = fitness(i); 
                Delta_pos = Positions(i,:);
            end
        end
        
        % --- GWO POSITION UPDATE ---
        a = 2 - l*((2)/Max_iter); 
        
        for i = 1:size(Positions, 1)
            for j = 1:size(Positions, 2)     
                r1 = rand(); r2 = rand();
                A1 = 2*a*r1 - a; C1 = 2*r2;
                D_alpha = abs(C1*Alpha_pos(j) - Positions(i,j));
                X1 = Alpha_pos(j) - A1*D_alpha;
                           
                r1 = rand(); r2 = rand();
                A2 = 2*a*r1 - a; C2 = 2*r2;
                D_beta = abs(C2*Beta_pos(j) - Positions(i,j));
                X2 = Beta_pos(j) - A2*D_beta;       
                
                r1 = rand(); r2 = rand();
                A3 = 2*a*r1 - a; C3 = 2*r2;
                D_delta = abs(C3*Delta_pos(j) - Positions(i,j));
                X3 = Delta_pos(j) - A3*D_delta;             
                
                Positions(i,j) = (X1 + X2 + X3) / 3;
            end
        end
        
        % --- GA CROSSOVER & MUTATION ---
        for i = 1:size(Positions, 1)
            if rand() < crossover_rate
                crossover_point = randi([1, dim]);
                Positions(i, crossover_point:end) = Alpha_pos(crossover_point:end);
            end
            
            if rand() < mutation_rate
                mutation_point = randi([1, dim]);
                noise = randn() * (ub(mutation_point) - lb(mutation_point)) * 0.1; 
                Positions(i, mutation_point) = Positions(i, mutation_point) + noise;
            end
        end
        
        Convergence_curve(l) = Alpha_score;
    end
    
    % Store the final results of this run
    all_runs_best_costs(current_run) = Alpha_score;
    all_runs_best_params(current_run, :) = Alpha_pos;
    all_convergence_curves(current_run, :) = Convergence_curve;
    
    fprintf('Run %d completed. Best Cost: %f\n', current_run, Alpha_score);
end

% =========================================================================
% 4. Statistical Analysis & Final Output
% =========================================================================

% Find the index of the absolute best run out of the 20
[absolute_best_cost, best_idx] = min(all_runs_best_costs);
absolute_best_parameters = all_runs_best_params(best_idx, :);

fprintf('\n==================================================\n');
fprintf('FINAL STATISTICAL RESULTS (Over %d Runs)\n', num_runs);
fprintf('==================================================\n');
fprintf('Best (Min) Cost   : %f\n', absolute_best_cost);
fprintf('Worst (Max) Cost  : %f\n', max(all_runs_best_costs));
fprintf('Mean Cost         : %f\n', mean(all_runs_best_costs));
fprintf('Standard Deviation: %f\n', std(all_runs_best_costs));
fprintf('--------------------------------------------------\n');
fprintf('Optimized Parameters (from the Best Run):\n');
fprintf('Nozzle Temp      : %f\n', absolute_best_parameters(1));
fprintf('Print Speed      : %f\n', absolute_best_parameters(2));
fprintf('Retraction Dist  : %f\n', absolute_best_parameters(3));
fprintf('Bed Temp         : %f\n', absolute_best_parameters(4));
fprintf('Flow Rate        : %f\n', absolute_best_parameters(5));
fprintf('==================================================\n');

% Plot the convergence curve of the ABSOLUTE BEST run
figure('Position', [500 500 660 290]);
plot(all_convergence_curves(best_idx, :), 'LineWidth', 2);
title(sprintf('Best GA-GWO Convergence Curve (Run %d)', best_idx));
xlabel('Iteration');
ylabel('Best Score (RMSE)');
grid on;

% =========================================================================
% Helper Functions
% =========================================================================

function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2); 
    if Boundary_no == 1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        Positions = zeros(SearchAgents_no, dim);
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end

function error = your_objective_function(agent_position, inputs, target)
    % Placeholder: Replace with actual ANFIS or prediction logic later
    error = sum((agent_position - mean(inputs, 1)).^2); 
end