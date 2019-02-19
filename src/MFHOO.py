
#Author: Rajat Sen

from __future__ import print_function
from __future__ import division

import numpy as np
from mf.mf_func import MFOptFunction  # MF function object
from utils.general_utils import map_to_cube # mapping everything to [0,1]^d cube
import sys
from examples.synthetic_functions import *
from multiprocessing import Process
#import brewer2mpl
import pandas as pd
import random
import sys
import time
import datetime




nu_mult = 1.0   # multiplier to the nu parameter

def flip(p):
	return True if random.random() < p else False

class MF_node(object):
	def __init__(self,cell,value,fidel,upp_bound,height,dimension,num):
		'''This is a node of the MFTREE
		cell: tuple denoting the bounding boxes of the partition
		m_value: mean value of the observations in the cell and its children
		value: value in the cell
		fidelity: the last fidelity that the cell was queried with
		upp_bound: B_{i,t} in the paper
		t_bound: upper bound with the t dependent term
		height: height of the cell (sometimes can be referred to as depth in the tree)
		dimension: the dimension of the parent that was halved in order to obtain this cell
		num: number of queries inside this partition so far
		left,right,parent: pointers to left, right and parent
		'''
		self.cell = cell
		self.m_value = value
		self.value = value
		self.fidelity = fidel
		self.upp_bound = upp_bound
		self.height = height
		self.dimension = dimension
		self.num = num
		self.t_bound = upp_bound

		self.left = None
		self.right = None
		self.parent = None

	def __cmp__(self,other):
		return cmp(other.t_bound,self.t_bound)


def in_cell(node,parent):
	'''
	Check if 'node' is a subset of 'parent'
	node can either be a MF_node or just a tuple denoting its cell
	'''
	try:
		ncell = list(node.cell)
	except:
		ncell = list(node)
	pcell = list(parent.cell)
	flag = 0

	for i in range(len(ncell)):
		if ncell[i][0] >= pcell[i][0] and ncell[i][1] <= pcell[i][1]:
			flag = 0
		else:
			flag = 1
			break
	if flag == 0:
		return True
	else:
		return False


class MF_tree(object):
	'''
	MF_tree class that maintains the multi-fidelity tree
	nu: nu parameter in the paper
	rho: rho parameter in the paper
	sigma: noise variance, ususally a hyperparameter for the whole process
	C: parameter for the bias function as defined in the paper
	root: can initialize a root node, when this parameter is supplied by a MF_node object instance
	'''
	def __init__(self,nu,rho,sigma,C,root = None):
		self.nu = nu
		self.rho = rho
		self.sigma = sigma
		self.root = root
		self.C = C 
		self.root = root
		self.mheight = 0
		self.maxi = float(-sys.maxsize - 1)
		self.current_best = root
	

	def insert_node(self,root,node):
		'''
		insert a node in the tree in the appropriate position
		'''
		if self.root is None:
			node.height = 0
			if self.mheight < node.height:
				self.mheight = node.height
			self.root = node
			self.root.parent = None
			return self.root
		if root is None:
			node.height = 0
			if self.mheight < node.height:
				self.mheight = node.height
			root = node
			root.parent = None
			return root
		if root.left is None and root.right is None:
			node.height = root.height + 1
			if self.mheight < node.height:
				self.mheight = node.height
			root.left = node
			root.left.parent = root
			return root.left
		elif root.left is not None:
			if in_cell(node,root.left):
				return self.insert_node(root.left,node)
			elif root.right is not None:
				if in_cell(node,root.right):
					return self.insert_node(root.right,node)
			else:
				node.height = root.height + 1
				if self.mheight < node.height:
					self.mheight = node.height
				root.right = node
				root.right.parent = root
				return root.right
	
	def update_parents(self,node,val):
		'''
		update the upperbound and mean value of a parent node, once a new child is inserted in its child tree. This process proceeds recursively up the tree
		'''
		if node.parent is None:
			return
		else:
			parent = node.parent
			parent.m_value = (parent.num*parent.m_value + val)/(1.0 + parent.num)
			parent.num = parent.num + 1.0
			parent.upp_bound = parent.m_value + 2*((self.rho)**(parent.height))*self.nu
			self.update_parents(parent,val)


	def update_tbounds(self,root,t):
		'''
		updating the tbounds of every node recursively
		'''
		if root is None:
			return
		self.update_tbounds(root.left,t)
		self.update_tbounds(root.right,t)
		root.t_bound = root.upp_bound + np.sqrt(2*(self.sigma**2)*np.log(t)/root.num)
		maxi = None
		if root.left:
			maxi = root.left.t_bound
		if root.right:
			if maxi:
				if maxi < root.right.t_bound:
					maxi = root.right.t_bound
			else:
				maxi = root.right.t_bound
		if maxi:
			root.t_bound = min(root.t_bound,maxi)


	def print_given_height(self,root,height):
		if root is None:
			return
		if root.height == height:
			print (root.cell, root.num,root.upp_bound,root.t_bound),
		elif root.height < height:
			if root.left:
				self.print_given_height(root.left,height)
			if root.right:
				self.print_given_height(root.right,height)
		else:
			return

	def levelorder_print(self):
		'''
		levelorder print
		'''
		for i in range(self.mheight + 1):
			self.print_given_height(self.root,i)
			print('\n')


	def search_cell(self,root,cell):
		'''
		check if a cell is present in the tree
		'''
		if root is None:
			return False,None,None
		if root.left is None and root.right is None:
			if root.cell == cell:
				return True,root,root.parent
			else:
				return False,None,root
		if root.left:
			if in_cell(cell,root.left):
				return self.search_cell(root.left,cell)
		if root.right:
			if in_cell(cell,root.right):
				return self.search_cell(root.right,cell)


	def get_next_node(self,root):
		'''
		getting the next node to be queried or broken, see the algorithm in the paper
		'''
		if root is None:
			print('Could not find next node. Check Tree.')
		if root.left is None and root.right is None:
			return root
		if root.left is None:
			return self.get_next_node(root.right)
		if root.right is None:
			return self.get_next_node(root.left)

		if root.left.t_bound > root.right.t_bound:
			return self.get_next_node(root.left)
		elif root.left.t_bound < root.right.t_bound:
			return self.get_next_node(root.right)
		else:
			bit = flip(0.5)
			if bit:
				return self.get_next_node(root.left)
			else:
				return self.get_next_node(root.right)


	def get_current_best(self,root):
		'''
		get current best cell from the tree
		'''
		if root is None:
			return
		if root.right is None and root.left is None:
			val = root.m_value - self.nu*((self.rho)**(root.height))
			if self.maxi < val:
				self.maxi = val 
				cell = list(root.cell) 
				self.current_best =np.array([(s[0]+s[1])/2.0 for s in cell])
			return
		if root.left:
			self.get_current_best(root.left)
		if root.right:
			self.get_current_best(root.right)









class MFHOO(object):
	'''
	MFHOO algorithm, given a fixed nu and rho
	mfobject: multi-fidelity noisy function object
	nu: nu parameter
	rho: rho parameter
	budget: total budget provided either in units or time in seconds
	sigma: noise parameter
	C: bias function parameter
	tol: default parameter to decide whether a new fidelity query is required for a cell
	Randomize: True implies that the leaf is split on a randomly chosen dimension, False means the scheme in DIRECT algorithm is used. We recommend using False.
	Auto: Select C automatically, which is recommended for real data experiments
	CAPITAL: 'Time' mean time in seconds is used as cost unit, while 'Actual' means unit cost used in synthetic experiments
	debug: If true then more messages are printed
	'''
	def __init__(self,mfobject, nu, rho, budget, sigma, C, tol = 1e-3,\
	 Randomize = False, Auto = False,value_dict = {},\
	 CAPITAL = 'Time', debug = 'True'):
		self.mfobject = mfobject
		self.nu = nu
		self.rho = rho
		self.budget = budget
		self.C = C
		self.t = 0
		self.sigma = sigma
		self.tol = tol
		self.Randomize = Randomize
		self.cost = 0
		self.cflag = False
		self.value_dict = value_dict
		self.CAPITAL = CAPITAL
		self.debug = debug
		if Auto:
			z1 = 0.8
			z2 = 0.2
			d = self.mfobject.domain_dim
			x = np.array([0.5]*d)
			t1 = time.time()
			v1 = self.mfobject.eval_at_fidel_single_point_normalised([z1], x)
			v2 = self.mfobject.eval_at_fidel_single_point_normalised([z2], x)
			t2 = time.time()
			self.C = np.sqrt(2)*np.abs(v1-v2)/np.abs(z1-z2)
			self.nu = nu_mult*self.C
			if self.debug:
				print('Auto Init: ')
				print('C: ' + str(self.C))
				print('nu: ' + str(self.nu))
			c1 = self.mfobject.eval_fidel_cost_single_point_normalised([z1])
			c2 = self.mfobject.eval_fidel_cost_single_point_normalised([z2])
			self.cost = c1 + c2
			if self.CAPITAL == 'Time':
				self.cost = t2 - t1
		d = self.mfobject.domain_dim
		cell = tuple([(0,1)]*d)
		height = 0
		dimension = 0
		root,cost= self.querie(cell,height, self.rho, self.nu, dimension, option = 1)
		self.t = self.t + 1
		self.Tree = MF_tree(nu,rho,self.sigma,C,root)
		self.Tree.update_tbounds(self.Tree.root,self.t)
		self.cost = self.cost + cost





	def get_value(self,cell,fidel):
		'''cell: tuple'''
		x = np.array([(s[0]+s[1])/2.0 for s in list(cell)])
		return self.mfobject.eval_at_fidel_single_point_normalised([fidel], x)

	def querie(self,cell,height, rho, nu,dimension,option = 1):
		diam = nu*(rho**height)
		if option == 1:
			z = min(max(1 - diam/self.C,self.tol),1.0)
		else:
			z = 1.0
		if cell in self.value_dict:
			current = self.value_dict[cell]
			if abs(current.fidelity - z) <= self.tol:
				value = current.value
				cost = 0
			else:
				t1 = time.time()
				value = self.get_value(cell,z)
				t2 = time.time()
				if abs(value - current.value) > self.C*abs(current.fidelity - z):
					self.cflag = True
				current.value = value
				current.m_value = value
				current.fidelity = z
				self.value_dict[cell] = current
				if self.CAPITAL == 'Time':
					cost = t2 - t1
				else:
					cost = self.mfobject.eval_fidel_cost_single_point_normalised([z])
		else:
			t1 = time.time()
			value = self.get_value(cell,z)
			t2 = time.time()
			bhi = 2*diam + value
			self.value_dict[cell] = MF_node(cell,value,z,bhi,height,dimension,1)
			if self.CAPITAL == 'Time':
				cost = t2 - t1
			else:
				cost = self.mfobject.eval_fidel_cost_single_point_normalised([z])

		bhi = 2*diam + value
		current_object = MF_node(cell,value,z,bhi,height,dimension,1)
		return current_object,cost


	def split_children(self,current,rho,nu,option = 1):
		pcell = list(current.cell)
		span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
		if self.Randomize:
			dimension = np.random.choice(range(len(pcell)))
		else:
			dimension = np.argmax(span)
		dd = len(pcell)
		if dimension == current.dimension:
			dimension = (current.dimension - 1)%dd
		cost = 0
		h = current.height + 1
		l = np.linspace(pcell[dimension][0],pcell[dimension][1],3)
		children = []
		for i in range(len(l)-1):
			cell = []
			for j in range(len(pcell)):
				if j != dimension:
					cell = cell + [pcell[j]]
				else:
					cell = cell + [(l[i],l[i+1])]
			cell = tuple(cell)
			child,c = self.querie(cell, h, rho, nu,dimension,option)
			children = children + [child]
			cost = cost + c

		return children, cost


	def take_HOO_step(self):
		current = self.Tree.get_next_node(self.Tree.root)
		children,cost = self.split_children(current,self.rho,self.nu,1)
		self.t = self.t + 2
		self.cost = self.cost + cost
		rnode = self.Tree.insert_node(self.Tree.root,children[0])
		self.Tree.update_parents(rnode,rnode.value)
		rnode = self.Tree.insert_node(self.Tree.root,children[1])
		self.Tree.update_parents(rnode,rnode.value)
		self.Tree.update_tbounds(self.Tree.root,self.t)


	def run(self):
		while self.cost <= self.budget:
			self.take_HOO_step()


	def get_point(self):
		self.Tree.get_current_best(self.Tree.root)
		return self.Tree.current_best




class MFPOO(object):
	'''
	MFPOO object that spawns multiple MFHOO instances
	'''
	def __init__(self,mfobject, nu_max, rho_max, total_budget, sigma, C, mult, tol = 1e-3, Randomize = False, Auto = False,unit_cost = 1.0, CAPITAL = 'Time', debug = 'True'):
		self.mfobject = mfobject
		self.nu_max = nu_max
		self.rho_max = rho_max
		self.total_budget = total_budget
		self.C = C
		self.t = 0
		self.sigma = sigma
		self.tol = tol
		self.Randomize = Randomize
		self.cost = 0
		self.value_dict = {}
		self.MH_arr = []
		self.CAPITAL = CAPITAL
		self.debug = debug
		if Auto:
			if unit_cost is None:
				z1 = 1.0
			else:
				z1 = 0.8
			z2 = 0.2
			d = self.mfobject.domain_dim
			x = np.array([0.5]*d)
			t1 = time.time()
			v1 = self.mfobject.eval_at_fidel_single_point_normalised([z1], x)
			t3 = time.time()
			v2 = self.mfobject.eval_at_fidel_single_point_normalised([z2], x)
			t2 = time.time()
			self.C = np.sqrt(2)*np.abs(v1-v2)/np.abs(z1-z2)
			self.nu_max = nu_mult*self.C
			if unit_cost is None:
				unit_cost = t3 - t1
				if self.debug:
					print('Setting unit cost automatically as None was supplied')
			if self.debug:
				print('Auto Init: ')
				print('C: ' + str(self.C))
				print('nu: ' + str(self.nu_max))
			c1 = self.mfobject.eval_fidel_cost_single_point_normalised([z1])
			c2 = self.mfobject.eval_fidel_cost_single_point_normalised([z2])
			self.total_budget = self.total_budget - c1 - c2
			if self.CAPITAL == 'Time':
				self.total_budget = self.total_budget - (t2 - t1)
			if self.debug:
				print('Budget Remaining: ' + str(self.total_budget))

		if self.CAPITAL == 'Time':
			self.unit_cost = unit_cost
		else:
			self.unit_cost = self.mfobject.eval_fidel_cost_single_point_normalised([1.0])
		n = max(self.total_budget/self.unit_cost,1)
		Dm = int(np.log(2.0)/np.log(1/self.rho_max))
		nHOO = int(mult*Dm*np.log(n/np.log(n+1)))
		self.nHOO = max(1,int(min(max(1,nHOO),n/2+1)))
		self.budget = (self.total_budget - self.nHOO*self.unit_cost)/float(self.nHOO)
		if self.debug:
			print('Number of MFHOO Instances: ' + str(self.nHOO))
			print('Budget per MFHOO Instance:' + str(self.budget))


	def run_all_MFHOO(self):
		nu = self.nu_max
		for i in range(self.nHOO):
			rho = self.rho_max**(float(self.nHOO)/(self.nHOO-i))
			MH = MFHOO(mfobject = self.mfobject, nu=nu, rho=rho, budget=self.budget, sigma=self.sigma, C=self.C, tol = 1e-3, Randomize = False, Auto = False,value_dict = self.value_dict, CAPITAL = self.CAPITAL, debug = self.debug)
			print('Running SOO number: ' + str(i+1) + ' rho: ' + str(rho) + ' nu: ' + str(nu))
			MH.run()
			print('Done!')
			self.cost = self.cost + MH.cost
			if MH.cflag:
				self.C = 1.4*self.C
				nu = nu_mult*self.C
				self.nu_max = nu_mult*self.C
				if self.debug:
					print('Updating C')
					print('C: ' + str(self.C))
					print('nu_max: ' + str(nu))
			self.value_dict = MH.value_dict
			self.MH_arr = self.MH_arr + [MH]


	def get_point(self):
		points = [H.get_point() for H in self.MH_arr]
		for H in self.MH_arr:
			self.t = self.t + H.t
		evals = [self.mfobject.eval_at_fidel_single_point_normalised([1.0],x) for x in points]
		if self.CAPITAL == 'Actual':
			self.cost = self.cost + self.nHOO*self.mfobject.eval_fidel_cost_single_point_normalised([1.0])
		else:
			self.cost = self.cost + self.nHOO*self.unit_cost
		
		index = np.argmax(evals)
		
		newp = []
		for p in points:
			_,npoint =  self.mfobject.get_unnormalised_coords(None,p)
			newp = newp + [npoint]
		
		return newp,evals





















