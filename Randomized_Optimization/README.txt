Randomized Optimization - Assignment 2 (https://github.com/davidcgong/ML_Coursework/tree/master/Randomized_Optimization)

This assignment was done using a modified version of ABAGAIL.

To get to where the code is and find a more user-friendly way of viewing this README, navigate to this link: https://github.com/davidcgong/ML_Coursework/tree/master/Randomized_Optimization

To run these algorithms, do the following steps. The only requirement is to have JDE and JRE installed or a way to run files through command line.

1) Navigate to where the Randomized_Optimization subfolder is
2) Running algorithms:

	1. Neural Network Tests
		-Command: 

			java -cp PATH project2.SpambaseTest spambase.arff ALG HN ITER -> {Additional parameters only if ALG = sa or ga}

		-For above:

			I.   PATH =  path to compiled java code. If not sure just substitute with ABAGAIL.jar
			II.  ALG =   randomized optimization algorithm {rhc, sa, ga} <- randomized hill climbing, simulated annealing, genetic algorithms
			III. HN =    # of hidden nodes
			IV.  ITER =  # of training iterations

		-Also note for additional parameters:

			I.   GA (Genetic Algorithms) has three which you need to add to the end in the following order {S MA MU}
				- S =  Starting Population
				- MA = # of Mating Individuals
				- MU = # of Mutations 
			II.  SA (Simulated Annealing) has two which you need to add to the end in the following order {ST CF}
				- ST = Starting Temperature
				- CF = Cooling Factor

			Each line prints out the sum of squared error and training accuracy for every iteration

		-Example (RHC with 20 hidden nodes and 25000 iterations):

			java -cp ABAGAIL.jar project2.SpambaseTest spambase.arff rhc 20 25000

	2. Optimization Problems

	Three were used (Traveling Salesman, Flip Flop, and Four Peaks).

		N = Input Size
		ITER = # of trials or times algorithm will be performed

		I. Traveling Salesman

			java -cp PATH project2.TravelingSalesmanTest N ITER

		II. Flip Flop

			java -cp PATH project2.FlipFlopTest N ITER

		III. Four Peaks

			java -cp PATH project2.FourPeaksTest N T ITER

			T represents the trigger point of the function.


		
		