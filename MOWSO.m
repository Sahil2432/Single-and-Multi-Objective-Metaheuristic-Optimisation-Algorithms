function [Pareto_Variables, Pareto_Costs] = MOWSO(SearchAgents_no, Max_iter, lb, ub, dim, fobj, Archive_max_size)
    % MOWSO Core Algorithm (No Hybridization)
    
    % WSO Specific Parameters
    f0 = 0.07;
    f1 = 0.0711;
    a0 = 6.250;
    a1 = 100;
    a2 = 0.0005;
    
    % Initialize Population
    empty_agent = struct('Position', [], 'Velocity', [], 'Cost', [], 'Best_Position', [], 'Best_Cost', []);
    population = repmat(empty_agent, SearchAgents_no, 1);
    
    for i = 1:SearchAgents_no
        population(i).Position = lb + (ub - lb) .* rand(1, dim);
        population(i).Velocity = zeros(1, dim);
        population(i).Cost = fobj(population(i).Position);
        population(i).Best_Position = population(i).Position;
        population(i).Best_Cost = population(i).Cost;
    end
    
    Archive = [];
    num_objectives = length(population(1).Cost);
    
    % Main Optimization Loop
    for iter = 1:Max_iter
        
        % 1. Update Pareto Archive
        for i = 1:SearchAgents_no
            is_dominated = false;
            for j = 1:length(Archive)
                if all(Archive(j).Cost <= population(i).Cost) && any(Archive(j).Cost < population(i).Cost)
                    is_dominated = true;
                    break;
                end
            end
            
            if ~is_dominated
                % Remove any archive members dominated by this new solution
                to_keep = true(1, length(Archive));
                for j = 1:length(Archive)
                    if all(population(i).Cost <= Archive(j).Cost) && any(population(i).Cost < Archive(j).Cost)
                        to_keep(j) = false;
                    end
                end
                Archive = Archive(to_keep);
                
                % Add to archive
                new_member.Position = population(i).Position;
                new_member.Cost = population(i).Cost;
                Archive = [Archive, new_member];
            end
        end
        
        % 2. Maintain Archive Size using Crowding Distance
        if length(Archive) > Archive_max_size
            costs = reshape([Archive.Cost], num_objectives, [])';
            distances = Calc_Crowding_Distance(costs);
            [~, sort_idx] = sort(distances, 'descend');
            Archive = Archive(sort_idx(1:Archive_max_size));
        end
        
        % 3. Calculate current archive distances for Leader Selection
        costs = reshape([Archive.Cost], num_objectives, [])';
        distances = Calc_Crowding_Distance(costs);
        
        % 4. WSO Movement Phase
        for i = 1:SearchAgents_no
            % Select prey (leader) from Archive using Tournament Selection
            if length(Archive) > 1
                idxs = randperm(length(Archive), 2);
                if distances(idxs(1)) > distances(idxs(2))
                    leader = Archive(idxs(1));
                else
                    leader = Archive(idxs(2));
                end
            else
                leader = Archive(1);
            end
            
            % WSO Dynamic Parameters
            p1 = a1 + (a2 - a1) * exp(-((4 * iter / Max_iter)^2));
            p2 = a0 + (a2 - a0) * exp(-((4 * iter / Max_iter)^2));
            mu = 2 / abs(2 - f0 - sqrt(f0^2 - 4*f0));
            
            % Velocity Update
            population(i).Velocity = mu * population(i).Velocity + ...
                rand() * p1 * (leader.Position - population(i).Position) + ...
                rand() * p2 * (population(i).Best_Position - population(i).Position);
                                     
            % Position Update
            new_pos = population(i).Position + population(i).Velocity / f1;
            
            % Enforce Boundaries
            new_pos = max(new_pos, lb);
            new_pos = min(new_pos, ub);
            
            % Evaluate
            new_cost = fobj(new_pos);
            
            % Update Personal Best
            if all(new_cost <= population(i).Best_Cost) && any(new_cost < population(i).Best_Cost)
                population(i).Best_Position = new_pos;
                population(i).Best_Cost = new_cost;
            elseif ~(all(population(i).Best_Cost <= new_cost) && any(population(i).Best_Cost < new_cost))
                if rand() < 0.5
                    population(i).Best_Position = new_pos;
                    population(i).Best_Cost = new_cost;
                end
            end
            
            population(i).Position = new_pos;
            population(i).Cost = new_cost;
        end
        
        disp(['Iteration ' num2str(iter) ' completed. Archive Size: ' num2str(length(Archive))]);
    end
    
    % Extract outputs
    Pareto_Variables = reshape([Archive.Position], dim, [])';
    Pareto_Costs = reshape([Archive.Cost], num_objectives, [])';
end

% --- Helper Function for Crowding Distance ---
function distances = Calc_Crowding_Distance(costs)
    n_sols = size(costs, 1);
    n_objs = size(costs, 2);
    distances = zeros(n_sols, 1);
    
    if n_sols <= 2
        distances(:) = inf; return;
    end
    
    for j = 1:n_objs
        [sorted_obj, sorted_idx] = sort(costs(:, j));
        distances(sorted_idx(1)) = inf;
        distances(sorted_idx(end)) = inf;
        
        obj_min = sorted_obj(1);
        obj_max = sorted_obj(end);
        
        if obj_max == obj_min, continue; end
        
        for i = 2:(n_sols - 1)
            distances(sorted_idx(i)) = distances(sorted_idx(i)) + ...
                (costs(sorted_idx(i+1), j) - costs(sorted_idx(i-1), j)) / (obj_max - obj_min);
        end
    end
end