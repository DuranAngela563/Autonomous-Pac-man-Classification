from __future__ import print_function
# bustersAgents.py
# ----------------

# Attribution Information: We have based our project on The Pacman AI projects that were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import os
from wekaI import Weka


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs




########   AUTOMATIC AGENT   ########

class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.countActions = 0
        self.weka = Weka()
        self.weka.start_jvm()

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

        filename = 'prueba.arff'
        existingFile = os.path.isfile('prueba.arff')
        self.file = open('prueba.arff', 'a')

        if not existingFile:
            # create header

            self.file.write('@relation prueba.arff\n')
            # self.file.write('@attribute Tick_Num numeric\n')
            self.file.write('@attribute Pacman_posx numeric\n')
            self.file.write('@attribute Pacman_posy numeric\n')

            self.file.write('@attribute Number_ghosts numeric\n')
            self.file.write('@attribute Ghost1_Distance numeric\n')
            self.file.write('@attribute Ghost2_Distance numeric\n')
            self.file.write('@attribute Ghost3_Distance numeric\n')
            self.file.write('@attribute Ghost4_Distance numeric\n')

            self.file.write('@attribute Ghost1_Posx numeric\n')
            self.file.write('@attribute Ghost1_Posy numeric\n')
            self.file.write('@attribute Ghost2_Posx numeric\n')
            self.file.write('@attribute Ghost2_Posy numeric\n')
            self.file.write('@attribute Ghost3_Posx numeric\n')
            self.file.write('@attribute Ghost3_Posy numeric\n')
            self.file.write('@attribute Ghost4_Posx numeric\n')
            self.file.write('@attribute Ghost4_Posy numeric\n')

            self.file.write('@attribute Living_Ghost1 {True,False}\n')
            self.file.write('@attribute Living_Ghost2 {True,False}\n')
            self.file.write('@attribute Living_Ghost3 {True,False}\n')
            self.file.write('@attribute Living_Ghost4 {True,False}\n')

            self.file.write('@attribute Ghost1_Direction {North, South, East,West, Stop, None}\n')
            self.file.write('@attribute Ghost2_Direction {North, South, East,West, Stop, None}\n')
            self.file.write('@attribute Ghost3_Direction {North, South, East,West, Stop, None}\n')
            self.file.write('@attribute Ghost4_Direction {North, South, East,West, Stop, None}\n')

            self.file.write('@attribute pacDots numeric\n')
            self.file.write('@attribute Dist_nearest_pacDots numeric\n')
            self.file.write('@attribute Pacman_direction {North, South, East,West, Stop}\n')
            self.file.write('@attribute Score numeric\n')

            self.file.write('@attribute East_Legal {True,False}\n')
            self.file.write('@attribute West_Legal {True,False}\n')
            self.file.write('@attribute North_Legal {True,False}\n')
            self.file.write('@attribute South_Legal {True,False}\n')

            self.file.write('@attribute FutureScore numeric\n')

            self.file.write('@data\n')

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):

        self.countActions = self.countActions + 1
        # self.printInfo(gameState)
        # print (self.printLineData(gameState))
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman

        # Save all variables
        Pacman_posx = gameState.getPacmanPosition()[0]
        Pacman_posy = gameState.getPacmanPosition()[1]
        Score = gameState.getScore()
        Ghost1_Distance = gameState.data.ghostDistances[0]
        Ghost3_Distance = gameState.data.ghostDistances[2]
        Ghost4_Distance = gameState.data.ghostDistances[3]
        Ghost1_Posx = gameState.getGhostPositions()[0][0]
        Ghost1_Posy = gameState.getGhostPositions()[0][1]
        Ghost2_Posx = gameState.getGhostPositions()[1][0]
        Ghost2_Posy = gameState.getGhostPositions()[1][1]
        Ghost3_Posx = gameState.getGhostPositions()[2][0]
        Ghost3_Posy = gameState.getGhostPositions()[2][1]
        Ghost4_Posx = gameState.getGhostPositions()[3][0]
        Ghost4_Posy = gameState.getGhostPositions()[3][1]
        pacDots = gameState.getNumFood()
        Dist_nearest_pacDots = gameState.getDistanceNearestFood()
        Ghost1_Direction = gameState.getGhostDirections().get(0)
        Ghost2_Direction = gameState.getGhostDirections().get(1)
        Ghost3_Direction = gameState.getGhostDirections().get(2)
        Ghost4_Direction = gameState.getGhostDirections().get(3)

        pacAc = gameState.getLegalPacmanActions()

        if 'North' in pacAc:
            North_Legal = True
        else:
            North_Legal = False
        if 'South' in pacAc:
            South_Legal = True
        else:
            South_Legal = False
        if 'West' in pacAc:
            West_Legal = True
        else:
            West_Legal = False
        if 'East' in pacAc:
            East_Legal = True
        else:
            East_Legal = False

        x = [Pacman_posx, Pacman_posy, Ghost1_Distance, Ghost3_Distance, Ghost4_Distance, Ghost1_Posx, Ghost1_Posy,
             Ghost2_Posx, Ghost2_Posy, Ghost3_Posx, Ghost3_Posy, Ghost4_Posx, Ghost4_Posy, Ghost1_Direction,
             Ghost2_Direction, Ghost3_Direction, Ghost4_Direction, pacDots, Dist_nearest_pacDots, Score, East_Legal,
             West_Legal, North_Legal, South_Legal]

        a = self.weka.predict('./j48model.model', x, './training_keyboard_model.arff')

        #####

        if (a == 'WEST') and Directions.WEST in legal:  move = Directions.WEST
        if (a == 'EAST') and Directions.EAST in legal: move = Directions.EAST
        if (a == 'NORTH') and Directions.NORTH in legal:   move = Directions.NORTH
        if (a == 'SOUTH') and Directions.SOUTH in legal: move = Directions.SOUTH

        return move


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):  ###########################
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

        self.countActions = 0
        self.line = ""

    def printLineData(self, gameState, move):
        # Our function printLineData returns all the information related to the current state.

        legal = gameState.getLegalActions(0)
        nearestfood = gameState.getDistanceNearestFood()
        if nearestfood == None:
            nearestfood = -1

        text = str(gameState.getPacmanPosition()[0]) + ',' \
               + str(gameState.getPacmanPosition()[1]) + ',' + str(gameState.getNumAgents() - 1) + \
               ',' + str(gameState.data.ghostDistances[0]) + ',' + str(gameState.data.ghostDistances[1]) + \
               ',' + str(gameState.data.ghostDistances[2]) + ',' + str(gameState.data.ghostDistances[3]) + \
               ',' + str(gameState.getGhostPositions()[0][0]) + ',' + str(gameState.getGhostPositions()[0][1]) + \
               ',' + str(gameState.getGhostPositions()[1][0]) + ',' + str(gameState.getGhostPositions()[1][1]) + \
               ',' + str(gameState.getGhostPositions()[2][0]) + ',' + str(gameState.getGhostPositions()[2][1]) + \
               ',' + str(gameState.getGhostPositions()[3][0]) + ',' + str(gameState.getGhostPositions()[3][1]) + \
               ',' + str(gameState.getLivingGhosts()[1]) + ',' + str(gameState.getLivingGhosts()[2]) + \
               ',' + str(gameState.getLivingGhosts()[3]) + ',' + str(gameState.getLivingGhosts()[4]) + \
               ',' + str(gameState.getGhostDirections().get(0)) + ',' + str(gameState.getGhostDirections().get(1)) + \
               ',' + str(gameState.getGhostDirections().get(2)) + ',' + str(gameState.getGhostDirections().get(3)) + \
               ',' + str(gameState.getNumFood()) + ',' + str(nearestfood) + ',' + str(move) + \
               ',' + str(gameState.getScore()) + ',' + str(Directions.EAST in legal) + ',' + str(
            Directions.WEST in legal) + \
               ',' + str(Directions.NORTH in legal) + ',' + str(Directions.SOUTH in legal)

        return text

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                gameState.data.ghostDistances[i] = 10000

        move = KeyboardAgent.getAction(self, gameState)

        # If we are not in Tick1
        if self.line != "":
            self.file.write(self.line + ", " + str(gameState.getScore()) + '\n')

        self.line = self.printLineData(gameState, move)

        return move


from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH

        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        '''


        # 1º get closest ghost
        min(gameState.data.ghostDistances)
        max_value = max(input_list)
        index = input_list.index(max_value)

        # choose action
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        '''

        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()

        # should have positions of living ghosts?
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]  # i+1 because i= 0 is the pacman
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

        filename = 'printLineData.arff'
        existingFile = os.path.isfile('printLineData.arff')
        self.file = open('printLineData.arff', 'a')

        if not existingFile:
            # create header

            self.file.write('@relation printLineData\n')
            self.file.write('@attribute Tick_Num numeric\n')
            self.file.write('@attribute Pacman_posx numeric\n')
            self.file.write('@attribute Pacman_posx numeric\n')

            self.file.write('@attribute Number_ghosts numeric\n')
            self.file.write('@attribute Ghost1_Distance numeric\n')
            self.file.write('@attribute Ghost2_Distance numeric\n')
            self.file.write('@attribute Ghost3_Distance numeric\n')
            self.file.write('@attribute Ghost4_Distance numeric\n')

            self.file.write('@attribute Ghost1_Posx numeric\n')
            self.file.write('@attribute Ghost1_Posy numeric\n')
            self.file.write('@attribute Ghost2_Posx numeric\n')
            self.file.write('@attribute Ghost2_Posy numeric\n')
            self.file.write('@attribute Ghost3_Posx numeric\n')
            self.file.write('@attribute Ghost3_Posy numeric\n')
            self.file.write('@attribute Ghost4_Posx numeric\n')
            self.file.write('@attribute Ghost4_Posy numeric\n')

            self.file.write('@attribute Living_Ghost1 {True,False}\n')
            self.file.write('@attribute Living_Ghost2 {True,False}\n')
            self.file.write('@attribute Living_Ghost3 {True,False}\n')
            self.file.write('@attribute Living_Ghost4 {True,False}\n')

            self.file.write('@attribute Ghost1_Direction {North, South, East,West, Stop}\n')
            self.file.write('@attribute Ghost2_Direction {North, South, East,West, Stop}\n')
            self.file.write('@attribute Ghost3_Direction {North, South, East,West, Stop}\n')
            self.file.write('@attribute Ghost4_Direction {North, South, East,West, Stop}\n')

            self.file.write('@attribute pacDots numeric\n')
            self.file.write('@attribute Dist_nearest_pacDots numeric\n')
            self.file.write('@attribute Pacman_direction {North, South, East,West, Stop}\n')
            self.file.write('@attribute Score numeric\n')

            self.file.write('@attribute East_Legal {True,False}\n')
            self.file.write('@attribute West_Legal {True,False}\n')
            self.file.write('@attribute North_Legal {True,False}\n')
            self.file.write('@attribute South_Legal {True,False}\n')

            self.file.write('@data\n')

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def printLineData(self, gameState, move):
        # file = open('lineData.txt', 'a')
        legal = gameState.getLegalActions(0)

        text = str(self.countActions) + ',' + str(gameState.getPacmanPosition()[0]) + ',' \
               + str(gameState.getPacmanPosition()[1]) + ',' + str(gameState.getNumAgents() - 1) + \
               ',' + str(gameState.data.ghostDistances[0]) + ',' + str(gameState.data.ghostDistances[1]) + \
               ',' + str(gameState.data.ghostDistances[2]) + ',' + str(gameState.data.ghostDistances[3]) + \
               ',' + str(gameState.getGhostPositions()[0][0]) + ',' + str(gameState.getGhostPositions()[0][1]) + \
               ',' + str(gameState.getGhostPositions()[1][0]) + ',' + str(gameState.getGhostPositions()[1][1]) + \
               ',' + str(gameState.getGhostPositions()[2][0]) + ',' + str(gameState.getGhostPositions()[2][1]) + \
               ',' + str(gameState.getGhostPositions()[3][0]) + ',' + str(gameState.getGhostPositions()[3][1]) + \
               ',' + str(gameState.getLivingGhosts()[0]) + ',' + str(gameState.getLivingGhosts()[1]) + \
               ',' + str(gameState.getLivingGhosts()[3]) + ',' + str(gameState.getLivingGhosts()[4]) + ',' + \
               ',' + str(gameState.getGhostDirections().get(0)) + ',' + str(gameState.getGhostDirections().get(1)) + \
               ',' + str(gameState.getGhostDirections().get(2)) + ',' + str(gameState.getGhostDirections().get(3)) + \
               ',' + str(gameState.getNumFood()) + ',' + str(gameState.getDistanceNearestFood()) + ',' + str(move) + \
               ',' + str(gameState.getScore()) + ',' + str(Directions.EAST in legal) + ',' + str(
            Directions.WEST in legal) + \
               ',' + str(Directions.NORTH in legal) + ',' + str(Directions.SOUTH in legal) + '\n'

        # elf.file.write(text)

        return text

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)

        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        prevMove = gameState.data.agentStates[0].getDirection()
        pacmanPosition = gameState.getPacmanPosition()

        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                gameState.data.ghostDistances[i] = 10000

            print('i: ', i)

        print('dist:', gameState.data.ghostDistances)
        minDistance = min(gameState.data.ghostDistances)
        minDistanceIndex = gameState.data.ghostDistances.index(minDistance)
        closerGhostPos = gameState.getGhostPositions()[minDistanceIndex]
        # print(gameState.data.ghostDistances)

        # We save the coordinates of the pacman and the nearest ghost for efficience
        pacx = pacmanPosition[0]
        pacy = pacmanPosition[1]
        ghostx = closerGhostPos[0]
        ghosty = closerGhostPos[1]

        print(pacx, pacy, ghostx, ghosty)
        # print(current_ghost)

        if (pacx < ghostx):

            if (prevMove != "West") and len(legal) > 2:
                if (Directions.EAST not in legal):
                    if (Directions.NORTH not in legal):
                        move = Directions.SOUTH
                    else:
                        move = Directions.NORTH
                if (Directions.EAST in legal):
                    move = Directions.EAST
            if (prevMove == "West") and len(legal) <= 2:
                move = Directions.EAST
            # if (prevMove == "West") and len(legal) == 2:
            #     move = Directions.NORTH

        if (pacx > ghostx):
            if (prevMove != "East") and len(legal) > 2:
                if (Directions.WEST not in legal):
                    if (Directions.SOUTH not in legal):
                        move = Directions.NORTH
                    else:
                        move = Directions.SOUTH
                if (Directions.WEST in legal):
                    move = Directions.WEST
            if (prevMove == "East") and len(legal) <= 2:
                move = Directions.WEST

        if (pacy < ghosty):
            if (prevMove != "South") and len(legal) > 2:
                if (Directions.NORTH not in legal):
                    if (Directions.WEST not in legal):
                        move = Directions.EAST
                    else:
                        move = Directions.EAST
                if (Directions.NORTH in legal):
                    move = Directions.NORTH
            if (prevMove == "South") and len(legal) <= 2:
                move = Directions.NORTH
            # if (prevMove == "South") and len(legal) == 2 and Directions.EAST not in legal:
            #     move = Directions.WEST
            # if (prevMove == "South") and len(legal) == 2 and Directions.WEST not in legal:
            #     move = Directions.EAST

        if (pacy > ghosty):
            if (prevMove != "North") and len(legal) > 2:
                if (Directions.SOUTH not in legal):
                    if (Directions.EAST not in legal):
                        move = Directions.WEST
                    else:
                        move = Directions.EAST
                if (Directions.SOUTH in legal):
                    move = Directions.SOUTH
            if (prevMove == "North") and len(legal) <= 2:
                move = Directions.SOUTH
            # if (prevMove == "North") and len(legal) == 2 and Directions.EAST not in legal:
            #     move = Directions.WEST
            # if (prevMove == "North") and len(legal) == 2 and Directions.WEST not in legal:
            #     move = Directions.EAST

        self.file.write(self.printLineData(gameState, move))

        return move

