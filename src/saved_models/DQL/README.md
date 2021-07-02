
# Model Descriptions

- DQL_30.06.2021-20_49.h5
  + 2 in, 2 out, no hidden layers
  + 500 iterations
  + trained on grid
  + reward: number of cars at tl  
  + yellow timeout 9s  


- DQL_01.07.2021-14:00.h5
  + 2 in, 2 out, 1 hidden layer with 10 nodes
  + 1000 iterations
  + trained on grid
  + reward: waiting time at tl (memory limit 100s)
  + yellow timeout 9s
    

- DQL_01.07.2021-20:00.h5
  + 2 in, 2 out, 2 hidden layers with 10 nodes each
  + 1500 iterations
  + trained on grid
  + reward: waiting time at tl (fixed memory problem)
  + yellow timeout 9s


- DQL_02.07.2021-10:52.h5
  + 2 in, 2 out, 2 hidden layers with 10 nodes each
  + 1500 iterations
  + trained on grid
  + reward: waiting time at tl (fixed memory problem)
  + yellow timeout 3s
  
- multi_DQL_02.07.2021-14:55.h5
  + 18 in, 2 out, 2 hidden layers with 10 nodes each
  + 5 iterations (just for testing)
  + trained on grid
  + reward: waiting time at tl (fixed memory problem)
  + yellow timeout 3s
  
- multi_DQL_02.07.2021-16:04.h5
  + 18 in, 2 out, 2 hidden layers with 10 nodes each
  + 1000 iterations
  + trained on grid
  + reward: waiting time at tl (fixed memory problem)
  + yellow timeout 3s