#Maximizing Drug Candidate Effectiveness with Bayesian Optimization

The task involves using Bayesian optimization to maximize a drug candidate's bioavailability (log) while ensuring it's easy to synthesize (SA < k). 
The optimization algorithm iterates through evaluating structural features (× subject to a synthetic accessibility constraint, aiming to find the best x* 
that maximizes log within the given constraints and noise in evaluations. More specifically, our goal was to minimize the normalized regret for not knowing the 
best hyperparameter value. In order to do so, we modified the traditional Bayesian optimization algorithm, 
GP_UCB, to take into account the SA constraint. Moreover we construct a function to select safe areas to which we limit sampling.
