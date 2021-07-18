# Traffree

Traffree aims to control traffic in order to reduce the waiting time for vehicles. Traffic jams remain a major problem in the daily lives of most people. With varying estimates, an average citizen wastes around 60-100 hours each year due to heavy traffic resulting in lost potential income, not saying anything about the extra damage we deal to the environment that makes the existing ecological situation even more worrying.

As it turns out suboptimal traffic signal controlling is one of the main reasons to blame, that’s why recently this problem became quite popular in machine learning applications. Standard traffic lights operate with alteration of predetermined phases that obviously can not deal with dynamic traffic flows throughout the day resulting in congestion in times when it could have been avoided in case of optimal control.

As a result we set a goal to create and compare traffic signalling models that would outperform standard performance in a simulated environment that closely resembled a real-life environment, resulting in reduced mean waiting time for vehicles.

We considered two approaches:
* traffic lights that can evaluate traffic condition only on connected roads and act accordingly;
* global model that gives us ability to observe the situation in the entire traffic network and decide what actions to take on each intersection.

For each approach we have experimented with models of different types and architectures out of which each one of them successfully outperformed standard traffic lights on test scenarios. Some of them significantly improved the average waiting time metric (mainly the models trained with evolutionary algorithm), yet there were some cases that fell short of our expectations. With the final results we were able to reduce the mean waiting time by 20%.

As the investigated area is currently under active research and only a limited number of scientific papers were applicable for our problem statement, thus we had to conduct large amounts of experiments and analyse the observed results. We think that our work is quite valuable and meaningful and coming from the importance of the topic, Traffree has a huge potential and plenty of areas for improvement.


## Getting Started

Installation is pretty straightforward, you just need to have conda and pull this project on your computer. 

Run this scripts to create and activate virtual environment: 
```
./bin/create-env.sh
```
```
conda activate ./env
```

That's it, you can successfully run this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
