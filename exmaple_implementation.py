from EvoModel import EvoModel, relu, softmax
import numpy as np
import gym
import sys
env = gym.make("CartPole-v1")

def error_func(y_test, y_predict):
	'''
	Using plain mean error
	'''
	return np.sum(np.abs(np.argmax(y_predict, axis = 1) - np.argmax(y_test, axis = 1)))

def cart_evaluator(nn_model, render=False):
	'''
	An evaluator to test the model
	'''
	observation = env.reset()
	done = False
	score = 0
	for _ in range(1000):
		if render:
			env.render()
		action = nn_model.predict(observation)
		observation, reward, done, info = env.step(np.argmax(action))
		score += reward
		if done:
			break

	return score

def main():

	#Args that are EvolutionaryBrain __init__ param
	model_args = {
		'input_size': 4,
		'shape': (4, 2),
		'middle_activation': relu,
		'final_activation': softmax,
		'epsilon': 5e-3
	}
	
	model = EvoModel(model_args, 250, cart_evaluator)
	for _ in range(40):
		model.train()
		model.eliminate(0.5)
		model.repopulate()
	
	cart_evaluator(model.nns[0][0], True)
	env.close()
	input()

if __name__ == '__main__':
	main()