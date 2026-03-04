import numpy as np
import yaml
from pathlib import Path
from gymnasium.spaces import Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleEconomyEnv(MultiAgentEnv):
    """
    Multi-agent economy simulation with:
    - Individual firm policies (heterogeneous learning)
    - Diversity bonus & market saturation penalty
    - Sequential purchasing with wealth-based preferences
    - Skill-based labor market
    - Quality-dependent production costs
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            default_config = yaml.safe_load(f)
        
        config = config or {}
        
        # Environment parameters
        env_cfg = default_config.get('environment', {})
        self.n_firms = config.get('n_firms', env_cfg.get('n_firms', 2))
        self.n_households = config.get('n_households', env_cfg.get('n_households', 10))
        self.max_steps = config.get('max_steps', env_cfg.get('max_steps', 100))
        
        # Initial ranges
        init_ranges = default_config.get('initial_ranges', {})
        self.init_ranges = {
            'firms': init_ranges.get('firms', {}),
            'households': init_ranges.get('households', {})
        }
        
        econ_cfg = default_config.get('economy', {})
        
        # Action adjustments (5 discrete levels: -10%, -5%, 0%, +5%, +10%)
        self.adjustment_rates = {0: -0.10, 1: -0.05, 2: 0.00, 3: 0.05, 4: 0.10}
        
        # Production
        prod_cfg = econ_cfg.get('production', {})
        self.productivity_base = prod_cfg.get('productivity_per_employee', 6.0)
        self.production_cost_per_unit = prod_cfg.get('cost_per_unit', 2.0)
        self.quality_cost_factor = prod_cfg.get('quality_cost_factor', 0.5)
        self.fixed_costs = prod_cfg.get('fixed_costs', 30.0)
        self.storage_cost_per_unit = prod_cfg.get('storage_cost_per_unit', 0.2)
        
        # Investment
        invest_cfg = econ_cfg.get('investment_costs', {})
        self.marketing_cost_per_level = invest_cfg.get('marketing_per_level', 20.0)
        self.quality_improvement_cost = invest_cfg.get('quality_improvement', 30.0)
        self.capacity_expansion_cost = invest_cfg.get('capacity_expansion', 50.0)
        
        # Bounds
        self.max_employees_hard_cap = econ_cfg.get('max_employees_hard_cap', 150)
        
        quality_bounds = econ_cfg.get('quality_bounds', {'min': 0.1, 'max': 1.0})
        marketing_bounds = econ_cfg.get('marketing_bounds', {'min': 0.1, 'max': 1.0})
        self.quality_min = quality_bounds['min']
        self.quality_max = quality_bounds['max']
        self.marketing_min = marketing_bounds['min']
        self.marketing_max = marketing_bounds['max']
        
        price_bounds = econ_cfg.get('price_bounds', {'min': 5.0, 'max': 100.0})
        wage_bounds = econ_cfg.get('wage_bounds', {'min': 5.0, 'max': 50.0})
        self.price_min = price_bounds['min']
        self.price_max = price_bounds['max']
        self.wage_min = wage_bounds['min']
        self.wage_max = wage_bounds['max']
        
        # Bankruptcy
        bankr_cfg = econ_cfg.get('bankruptcy', {})
        self.bankruptcy_threshold = bankr_cfg.get('threshold', -400.0)
        self.bankruptcy_penalty = bankr_cfg.get('penalty_reward', -20.0)
        
        # Households
        hh_cfg = econ_cfg.get('households', {})
        self.consumption_rate = hh_cfg.get('consumption_rate', 0.7)
        self.utility_price_weight = hh_cfg.get('utility_price_weight', 1.0)
        self.utility_quality_weight = hh_cfg.get('utility_quality_weight', 0.5)
        self.utility_marketing_weight = hh_cfg.get('utility_marketing_weight', 0.3)
        
        wealth_util_mods = hh_cfg.get('wealth_utility_modifiers', {})
        self.wealth_utility_modifiers = {
            'low': wealth_util_mods.get('low', {'price_weight': 1.0, 'quality_weight': 1.0, 'marketing_weight': 1.0}),
            'medium': wealth_util_mods.get('medium', {'price_weight': 1.0, 'quality_weight': 1.0, 'marketing_weight': 1.0}),
            'high': wealth_util_mods.get('high', {'price_weight': 1.0, 'quality_weight': 1.0, 'marketing_weight': 1.0})
        }
        
        wealth_mult = hh_cfg.get('wealth_multipliers', {})
        self.wealth_multipliers = {
            'low': wealth_mult.get('low', 0.8),
            'medium': wealth_mult.get('medium', 1.0),
            'high': wealth_mult.get('high', 1.2)
        }
        
        # Skills
        skill_cfg = econ_cfg.get('skill_system', {})
        self.skill_base_multiplier = skill_cfg.get('base_multiplier', 0.5)
        self.skill_factor = skill_cfg.get('skill_factor', 0.5)
        
        # Rewards
        reward_cfg = econ_cfg.get('reward', {})
        self.reward_scale = reward_cfg.get('scale', 100.0)
        self.capital_bonus_divisor = reward_cfg.get('capital_bonus_divisor', 1000.0)
        self.capital_bonus_max = reward_cfg.get('capital_bonus_max', 5.0)
        self.market_share_bonus_weight = reward_cfg.get('market_share_bonus', 10.0)
        self.inventory_penalty_weight = reward_cfg.get('inventory_penalty', 2.0)
        self.exploration_penalty = reward_cfg.get('exploration_penalty', 5.0)
        self.survivor_diversity_threshold = reward_cfg.get('survivor_diversity_threshold', 5)
        self.survivor_diversity_penalty = reward_cfg.get('survivor_diversity_penalty', 10000)
        self.diversity_bonus_weight = reward_cfg.get('diversity_bonus_weight', 50.0)
        self.saturation_penalty_base = reward_cfg.get('saturation_penalty_base', 100.0)
        self.saturation_optimal_per_segment = reward_cfg.get('saturation_optimal_per_segment', 3)
        
        self._agent_ids = set(f"firm_{i}" for i in range(self.n_firms))
        self._obs_space = Box(low=-1000.0, high=1000.0, shape=(25,), dtype=np.float32)
        self._action_space = MultiDiscrete([5, 5, 3, 2, 3])  # price, wage, marketing, quality, capacity
        
        self.reset()
    
    @property
    def observation_space(self):
        return self._obs_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        firm_ranges = self.init_ranges.get('firms', {})
        hh_ranges = self.init_ranges.get('households', {})
        
        price_range = firm_ranges.get('price', {'min': 20.0, 'max': 40.0})
        wage_range = firm_ranges.get('wage', {'min': 15.0, 'max': 30.0})
        emp_range = firm_ranges.get('max_employees', {'min': 3, 'max': 10})
        capital_range = firm_ranges.get('capital', {'min': 1000.0, 'max': 1500.0})
        quality_range = firm_ranges.get('quality', {'min': 0.5, 'max': 0.8})
        marketing_range = firm_ranges.get('marketing', {'min': 0.3, 'max': 0.6})
        
        money_range = hh_ranges.get('money', {'min': 100.0, 'max': 200.0})
        skill_range = hh_ranges.get('skill_level', {'min': 0.3, 'max': 1.0})
        max_price_range = hh_ranges.get('max_acceptable_price', {'min': 10.0, 'max': 100.0})
        wealth_dist = hh_ranges.get('wealth_distribution', {'low': 0.3, 'medium': 0.5, 'high': 0.2})
        
        # Initialize firms
        self.firms = {}
        for i in range(self.n_firms):
            self.firms[f"firm_{i}"] = {
                'price': round(np.random.uniform(price_range['min'], price_range['max']), 2),
                'wage': round(np.random.uniform(wage_range['min'], wage_range['max']), 2),
                'max_employees': np.random.randint(emp_range['min'], emp_range['max'] + 1),
                'employees': 0,
                'inventory': 0,
                'production': 0,
                'capital': round(np.random.uniform(capital_range['min'], capital_range['max']), 2),
                'profit': 0.0,
                'profit_last_step': 0.0,
                'revenue': 0.0,
                'costs': 0.0,
                'sales': 0,
                'sales_last_step': 0,
                'quality': round(np.random.uniform(quality_range['min'], quality_range['max']), 2),
                'marketing': round(np.random.uniform(marketing_range['min'], marketing_range['max']), 2),
                'bankrupt': False,
            }
        
        # Initialize households
        wealth_probs = [wealth_dist['low'], wealth_dist['medium'], wealth_dist['high']]
        self.households = []
        for _ in range(self.n_households):
            self.households.append({
                'money': round(np.random.uniform(money_range['min'], money_range['max']), 2),
                'skill_level': round(np.random.uniform(skill_range['min'], skill_range['max']), 2),
                'max_acceptable_price': round(np.random.uniform(max_price_range['min'], max_price_range['max']), 2),
                'employer': None,
                'wage': 0.0,
                'wealth_type': np.random.choice(['low', 'medium', 'high'], p=wealth_probs),
            })
        
        self.timestep = 0
        self.bankruptcies_this_episode = 0
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        return obs, {agent_id: {} for agent_id in self._agent_ids}
    
    def _classify_segment(self, firm):
        """Classify firm into market segment based on price/quality"""
        price = firm['price']
        quality = firm['quality']
        
        if price < 40:
            return 'budget'
        elif price > 70 or quality > 1.5:
            return 'premium'
        else:
            return 'mainstream'
    
    def _calculate_diversity_bonus(self, firm_id):
        """Reward for unique strategy (Euclidean distance to nearest competitor)"""
        firm = self.firms[firm_id]
        my_strategy = np.array([firm['price'], firm['quality'], firm['marketing']])
        
        min_distance = float('inf')
        
        for other_id, other_firm in self.firms.items():
            if other_id == firm_id or other_firm['bankrupt']:
                continue
            
            other_strategy = np.array([other_firm['price'], other_firm['quality'], other_firm['marketing']])
            distance = np.linalg.norm(my_strategy - other_strategy)
            min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 0.0
        
        normalized_distance = min(min_distance / 100.0, 1.0)
        return normalized_distance * self.diversity_bonus_weight
    
    def _calculate_saturation_penalty(self, firm_id):
        """Penalize overcrowded market segments"""
        firm = self.firms[firm_id]
        my_segment = self._classify_segment(firm)
        
        segment_counts = {'budget': 0, 'mainstream': 0, 'premium': 0}
        for other_id, other_firm in self.firms.items():
            if not other_firm['bankrupt']:
                segment_counts[self._classify_segment(other_firm)] += 1
        
        firms_in_segment = segment_counts[my_segment]
        
        if firms_in_segment <= self.saturation_optimal_per_segment:
            return 0.0
        
        overcrowding = firms_in_segment - self.saturation_optimal_per_segment
        penalty = -overcrowding * self.saturation_penalty_base
        
        if firms_in_segment > 5:
            penalty *= 2
        
        return penalty
    
    def step(self, action_dict):
        active_firms = [aid for aid in self._agent_ids if not self.firms[aid]['bankrupt']]
        
        # Phase 1: Firms take actions
        for agent_id in active_firms:
            action = action_dict.get(agent_id, [2, 2, 1, 0, 1])
            firm = self.firms[agent_id]
            
            # Price adjustment
            price_adj = self.adjustment_rates[action[0]]
            if price_adj != 0:
                firm['price'] *= (1 + price_adj)
            firm['price'] = round(np.clip(firm['price'], self.price_min, self.price_max), 2)
            
            # Wage adjustment
            wage_adj = self.adjustment_rates[action[1]]
            if wage_adj != 0:
                firm['wage'] *= (1 + wage_adj)
            firm['wage'] = round(np.clip(firm['wage'], self.wage_min, self.wage_max), 2)
            
            # Marketing
            if action[2] == 0:
                firm['marketing'] = max(self.marketing_min, firm['marketing'] - 0.1)
            elif action[2] == 2:
                if firm['capital'] >= self.marketing_cost_per_level:
                    firm['capital'] = round(firm['capital'] - self.marketing_cost_per_level, 2)
                    firm['marketing'] = min(self.marketing_max, firm['marketing'] + 0.1)
            firm['marketing'] = round(firm['marketing'], 2)
            
            # Quality
            if action[3] == 1:
                if firm['capital'] >= self.quality_improvement_cost:
                    firm['capital'] = round(firm['capital'] - self.quality_improvement_cost, 2)
                    firm['quality'] = min(self.quality_max, firm['quality'] + 0.05)
            firm['quality'] = round(firm['quality'], 2)
            
            # Capacity
            if action[4] == 0:
                firm['max_employees'] = max(1, firm['max_employees'] - 1)
            elif action[4] == 2:
                if firm['capital'] >= self.capacity_expansion_cost and firm['max_employees'] < self.max_employees_hard_cap:
                    firm['capital'] = round(firm['capital'] - self.capacity_expansion_cost, 2)
                    firm['max_employees'] += 1
            
            firm['employees'] = 0
            firm['employee_skills'] = []
            firm['sales_last_step'] = firm['sales']
            firm['sales'] = 0
        
        # Phase 2: Labor market (skill-based matching)
        for household in self.households:
            household['employer'] = None
            household['wage'] = 0.0
        
        households_sorted = sorted(self.households, key=lambda h: h['skill_level'], reverse=True)
        firms_by_wage = sorted(active_firms, key=lambda aid: self.firms[aid]['wage'], reverse=True)
        
        for household in households_sorted:
            for firm_id in firms_by_wage:
                firm = self.firms[firm_id]
                if firm['employees'] < firm['max_employees']:
                    household['employer'] = firm_id
                    household['wage'] = firm['wage']
                    firm['employees'] += 1
                    firm['employee_skills'].append(household['skill_level'])
                    household['money'] = round(household['money'] + firm['wage'], 2)
                    firm['capital'] = round(firm['capital'] - firm['wage'], 2)
                    break
        
        # Unemployed work for suppliers
        if active_firms:
            wages = [self.firms[aid]['wage'] for aid in active_firms]
            supplier_wage = round((min(wages) + max(wages)) / 2.0, 2)
            
            for household in self.households:
                if household['employer'] is None:
                    household['employer'] = 'suppliers'
                    household['wage'] = supplier_wage
                    household['money'] = round(household['money'] + supplier_wage, 2)
        
        # Phase 3: Production
        for agent_id in active_firms:
            firm = self.firms[agent_id]
            
            if firm['employees'] > 0:
                avg_skill = np.mean(firm['employee_skills'])
                skill_multiplier = self.skill_base_multiplier + (avg_skill * self.skill_factor)
                production = int(firm['employees'] * self.productivity_base * skill_multiplier)
                firm['production'] = production
                firm['inventory'] += production
            else:
                firm['production'] = 0
        
        # Phase 4: Sequential purchasing (wealth-based utility)
        total_sales = {agent_id: 0 for agent_id in active_firms}
        households_random = self.households.copy()
        np.random.shuffle(households_random)
        
        for household in households_random:
            if household['money'] <= 0:
                continue
            
            budget = household['money'] * self.consumption_rate
            budget *= self.wealth_multipliers.get(household['wealth_type'], 1.0)
            remaining_budget = budget
            
            max_price = household['max_acceptable_price']
            affordable_firms = [fid for fid in active_firms if self.firms[fid]['price'] <= max_price]
            
            if not affordable_firms:
                continue
            
            # Calculate utility
            wealth_type = household['wealth_type']
            wealth_mods = self.wealth_utility_modifiers.get(wealth_type, {})
            
            price_weight = self.utility_price_weight * wealth_mods.get('price_weight', 1.0)
            quality_weight = self.utility_quality_weight * wealth_mods.get('quality_weight', 1.0)
            marketing_weight = self.utility_marketing_weight * wealth_mods.get('marketing_weight', 1.0)
            
            utilities = {}
            for firm_id in affordable_firms:
                firm = self.firms[firm_id]
                if firm['price'] > 0:
                    numerator = (firm['quality'] * quality_weight + firm['marketing'] * marketing_weight)
                    denominator = firm['price'] * price_weight
                    utilities[firm_id] = numerator / denominator
                else:
                    utilities[firm_id] = 0.0
            
            if not utilities:
                continue
            
            firms_sorted_by_utility = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
            
            for firm_id, _ in firms_sorted_by_utility:
                if remaining_budget <= 0:
                    break
                
                firm = self.firms[firm_id]
                if firm['inventory'] <= 0:
                    continue
                
                max_qty_budget = remaining_budget / firm['price'] if firm['price'] > 0 else 0
                max_qty_inventory = firm['inventory']
                quantity = int(min(max_qty_budget, max_qty_inventory))
                
                if quantity > 0:
                    actual_cost = round(quantity * firm['price'], 2)
                    total_sales[firm_id] += quantity
                    household['money'] = round(household['money'] - actual_cost, 2)
                    firm['inventory'] -= quantity
                    remaining_budget = round(remaining_budget - actual_cost, 2)
        
        for agent_id in active_firms:
            self.firms[agent_id]['sales'] = total_sales.get(agent_id, 0)
        
        # Phase 5: Calculate rewards
        rewards = {}
        total_market_sales = sum(self.firms[aid]['sales'] for aid in active_firms)
        
        for agent_id in self._agent_ids:
            firm = self.firms[agent_id]
            
            if firm['bankrupt']:
                rewards[agent_id] = self.bankruptcy_penalty
                continue
            
            revenue = round(total_sales.get(agent_id, 0) * firm['price'], 2)
            
            quality_cost_multiplier = 1.0 + (firm['quality'] * self.quality_cost_factor)
            production_costs = round(firm['production'] * self.production_cost_per_unit * quality_cost_multiplier, 2)
            storage_costs = round(firm['inventory'] * self.storage_cost_per_unit, 2)
            total_costs = round(production_costs + storage_costs + self.fixed_costs, 2)
            
            profit = round(revenue - total_costs, 2)
            
            firm['revenue'] = revenue
            firm['costs'] = total_costs
            firm['profit_last_step'] = firm['profit']
            firm['profit'] = profit
            firm['capital'] = round(firm['capital'] + profit, 2)
            
            # Check bankruptcy
            if firm['capital'] < self.bankruptcy_threshold:
                firm['bankrupt'] = True
                self.bankruptcies_this_episode += 1
                rewards[agent_id] = self.bankruptcy_penalty
            else:
                base_reward = profit / self.reward_scale
                
                capital_bonus = 0.0
                if firm['capital'] > 0:
                    capital_bonus = min(firm['capital'] / self.capital_bonus_divisor, self.capital_bonus_max)
                
                market_share_bonus = 0.0
                if total_market_sales > 0:
                    own_share = firm['sales'] / total_market_sales
                    market_share_bonus = own_share * self.market_share_bonus_weight
                
                inventory_penalty = 0.0
                if firm['production'] > 0:
                    inv_ratio = firm['inventory'] / firm['production']
                    if inv_ratio > 1.0:
                        inventory_penalty = -inv_ratio * self.inventory_penalty_weight
                
                exploration_penalty = -self.exploration_penalty if revenue == 0 else 0.0
                
                diversity_bonus = self._calculate_diversity_bonus(agent_id)
                saturation_penalty = self._calculate_saturation_penalty(agent_id)
                
                rewards[agent_id] = (
                    base_reward + capital_bonus + market_share_bonus + 
                    inventory_penalty + exploration_penalty + diversity_bonus + saturation_penalty
                )
        
        # Survivor diversity penalty (episode end)
        if self.timestep == self.max_steps - 1:
            active_count = sum(1 for f in self.firms.values() if not f['bankrupt'])
            if active_count < self.survivor_diversity_threshold:
                missing = self.survivor_diversity_threshold - active_count
                penalty = missing * self.survivor_diversity_penalty
                for agent_id in self._agent_ids:
                    if not self.firms[agent_id]['bankrupt']:
                        rewards[agent_id] -= penalty
        
        self.timestep += 1
        done = self.timestep >= self.max_steps or all(self.firms[aid]['bankrupt'] for aid in self._agent_ids)
        
        obs = {agent_id: self._get_obs(agent_id) for agent_id in self._agent_ids}
        dones = {agent_id: done for agent_id in self._agent_ids}
        dones['__all__'] = done
        
        infos = {
            agent_id: {
                'profit': self.firms[agent_id]['profit'],
                'revenue': self.firms[agent_id]['revenue'],
                'costs': self.firms[agent_id]['costs'],
                'capital': self.firms[agent_id]['capital'],
                'employees': self.firms[agent_id]['employees'],
                'inventory': self.firms[agent_id]['inventory'],
                'production': self.firms[agent_id]['production'],
                'sales': self.firms[agent_id]['sales'],
                'quality': self.firms[agent_id]['quality'],
                'marketing': self.firms[agent_id]['marketing'],
                'bankrupt': self.firms[agent_id]['bankrupt'],
            } 
            for agent_id in self._agent_ids
        }
        
        return obs, rewards, dones, dones, infos
    
    def _get_obs(self, agent_id):
        """Create observation vector (25 features)"""
        firm = self.firms[agent_id]
        active_firms = [f for aid, f in self.firms.items() if not f['bankrupt']]
        
        if not active_firms:
            return np.zeros(25, dtype=np.float32)
        
        all_prices = [f['price'] for f in active_firms]
        all_wages = [f['wage'] for f in active_firms]
        
        market_avg_price = np.mean(all_prices)
        market_min_price = np.min(all_prices)
        market_max_price = np.max(all_prices)
        market_avg_wage = np.mean(all_wages)
        market_min_wage = np.min(all_wages)
        market_max_wage = np.max(all_wages)
        
        total_employed = sum(f['employees'] for f in active_firms)
        unemployment_rate = 1.0 - (total_employed / self.n_households)
        avg_household_money = np.mean([hh['money'] for hh in self.households])
        avg_household_skill = np.mean([hh['skill_level'] for hh in self.households])
        competitors_alive = len(active_firms)
        
        total_market_sales = sum(f['sales'] for f in active_firms)
        own_market_share = firm['sales'] / total_market_sales if total_market_sales > 0 else 1.0 / len(active_firms)
        sales_trend = firm['sales'] - firm['sales_last_step']
        inventory_ratio = firm['inventory'] / firm['production'] if firm['production'] > 0 else 0.0
        
        market_median_wage = np.median(all_wages)
        market_median_price = np.median(all_prices)
        wage_competitiveness = firm['wage'] / market_median_wage if market_median_wage > 0 else 1.0
        price_competitiveness = firm['price'] / market_median_price if market_median_price > 0 else 1.0
        
        obs = np.array([
            firm['price'], firm['wage'], firm['employees'], firm['inventory'], firm['capital'] / 100.0,
            firm['quality'], firm['marketing'],
            market_avg_price, market_min_price, market_max_price,
            market_avg_wage, market_min_wage, market_max_wage,
            total_employed, unemployment_rate, avg_household_money / 10.0, avg_household_skill,
            firm['profit_last_step'] / self.reward_scale, competitors_alive, self.timestep / self.max_steps,
            own_market_share, sales_trend / 10.0, inventory_ratio,
            wage_competitiveness, price_competitiveness,
        ], dtype=np.float32)
        
        return np.clip(obs, -1000.0, 1000.0)
