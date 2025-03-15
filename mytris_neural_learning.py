###############################################################################
# Program: mytris_neural_learning                
# Rev: 2.4                                       
# Updated: 15th of March 2025                    
# Goal: Tris (Tic Tac Toe) game using neural networks and automatic learning.
#
# MIT License
#
# Copyright (c) 2025 Marco Mattiucci
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################




import random
import os.path




CIRCLE = 1  # Computer symbol O
STAR = -1   # Human/User player symbol X
EMPTY = 0   # Empty cell _




### Class for single perceptron node
####################################
class myPerceptron: 

    #ATTRIBUTES(myPerceptron):
    ##########################
    
    perceptron_name = None  # Name of the node
    trigger_level = None    # Perceptron activation trigger (usually float value 0.9)
    status = None           # Perceptron internal status (float value - for an activated node it is 1.0)
                            # the approach is simplified so the status is also the output of the node.

    #METHODS(myPerceptron):
    #######################

    ##############################################################################
    ### The myPerceptron constructor just defines some internal values of the node
    ##############################################################################
    def __init__(self,name,trigger_level = 0.9): 
        self.perceptron_name = name         # define perceptron name
        self.trigger_level = trigger_level  # define trigger
        self.status = EMPTY                 # define the initial status as EMPTY (0)




### Class for a perceptron network
##################################
class myPerceptronNetwork:

    #ATTRIBUTES(myPerceptronNetwork):
    #################################

    MAX_NR_OF_NODES = None      # max number of perceptron nodes
    network_name = None         # name of the perceptron network
    network_dimension = None    # network dimension (number of nodes already initialized)
    perceptron_nodes = None     # list of initialized perceptron nodes
    links_and_weights = None    # sparse matrix defining the weighted links between pairs of perceptron nodes
    weights_0 = None            # weights associated with nodes even without links

    #METHODS(myPerceptronNetwork):
    ##############################

    ########################################################################################    
    ### The myPerceptronNetwork constructor defines the basic internal values of the network
    ########################################################################################
    def __init__(self,name,max_nr_of_nodes):
        self.MAX_NR_OF_NODES = int(max_nr_of_nodes) # define the max number of allowed perceptrons
        self.network_name = name                    # define the perceptron network name
        self.network_dimension = 0                  # define the starting network dimensions as 0
        self.perceptron_nodes = list()              # initialise the initilized perceptron node list
        # allocate the sparse matrix with no links and no weights:
        self.links_and_weights = [[None for _ in range(self.MAX_NR_OF_NODES)] for _ in range(self.MAX_NR_OF_NODES)]
        # allocate the list of weights associated with nodes even without links:
        self.weights_0 = [None for _ in range(self.MAX_NR_OF_NODES)]    

    #########################################################################################################
    ### myPerceptronNetwork "new_node" method creates a new perceptron node and inserts it into the node list
    #########################################################################################################
    def new_node(self):     
        self.perceptron_nodes.append(myPerceptron("Node"+str(self.network_dimension)))  # insert the new node in the "perceptron_nodes" list:
        self.network_dimension += 1         # increase the network dimension
        return self.network_dimension-1     # return the integer (id) that identifies the new node

    ###########################################################################################################
    ### myPerceptronNetwork "new_link" method establishes a new weight-oriented link between 2 existing nodes
    ###########################################################################################################
    def new_link(self,from_node_id,to_node_id,weight):  
        # check if the 2 node IDs are good (between 0 and initialized network dimension):
        if from_node_id >= 0 and from_node_id < self.network_dimension and to_node_id >= 0 and to_node_id < self.network_dimension:
            self.links_and_weights[from_node_id][to_node_id] = weight   # set the link and the related weight in the sparse matrix
            return True                                                 # return success
        else:
            print("ERROR 1 from class myPerceptronNetwork: bad node id(s) [",from_node_id,",",to_node_id,"]")
            quit()

    ##################################################################################################
    ### myPerceptronNetwork "node_inputs" method defines the inputs of "to_node_id" by a "input_list"
    ### "input_list" is made of couples (from_node_id,weight) where "from_node_id" is the input node
    ### and "weight" is the value (float) of the link that will be established.
    ##################################################################################################
    def node_inputs(self,to_node_id,input_list):    
        for (from_node_id,weight) in input_list:    # for every "from_node_id" in the "input_list" create a new link towards "to_node_id"
            if from_node_id == None:                # if "from_node_id" is None, it means we're dealing with the weight that is not related to any link
                self.weights_0[to_node_id] = weight # set the value of the weight that is not related to any link
                continue                            # go on with all the other input links
            if not self.new_link(from_node_id,to_node_id,weight):   # build the link "from_node_id" - "to_node_id" by "weight" (float)
                # If the function returns a value other than True, an error has occurred:
                print("ERROR 2 from class myPerceptronNetwork: bad node inputs [",input_list,",",to_node_id,"]")
                quit()

    ############################################################################
    ### myPerceptronNetwork method that defines the new status of a node based on      
    ### all its inputs. The inputs of a node are the values of the status of   
    ### every linked input node multiplied by the related weight.                                   
    ### Be aware that the internal "status" of a node is also its output.               
    ############################################################################
    def evaluate_new_node_status(self,node_id):     # Evaluate the subsequent state of "node_id"
        input_values = list()                       # Initialize the list of input values
        for j in range(self.network_dimension):     # Find all nodes that are connected as input to "node_id"
            if self.links_and_weights[j][node_id] != None:  # There is a link if the sparse matrix is ​​not None
                input_values.append((self.links_and_weights[j][node_id],j)) # In that case collect the weight that connects the two nodes
        acc = 0.0                                   # Initialize the internal node function calculation
        for (w,j) in input_values:                  # Sum all input values ​​multiplied by the related weight
            acc += w * self.perceptron_nodes[j].status
        if self.weights_0[node_id] != None:         # If there is a weight not related to any link, add it
            acc += self.weights_0[node_id]
        if acc > self.perceptron_nodes[node_id].trigger_level:  # Activation function: if the sum is greater than the trigger value
            if node_id < 9:                         # The activation for the nodes of the game board set a CIRCLE in the related cell.
                if self.perceptron_nodes[node_id].status == EMPTY:      # only an empty cell can be written
                    self.perceptron_nodes[node_id].status = CIRCLE   # CIRCLE in the cell
                    return "move_done"  # return that a move has been done
                else:
                    return "no_status_change"   # otherwise return that nothing has been done
            else:
                self.perceptron_nodes[node_id].status = CIRCLE      # For all the other perceptrons the activation status is CIRCLE, so 1.0
                return "activated_node"                             # return that "node_id" has been activated
        else:
            return "non_activated_node" # return that "node_id" hasn't been activated

    #################################################################################
    ### myPerceptronNetwork Method that evaluates the next status of a list of nodes 
    ### sequentially, from the first to the last, without caring about the results.
    ### The input is "node_id_set", a list of node ids.
    #################################################################################
    def evaluate_new_status_for_all_nodes_sequentially(self,node_id_set): 
        for node_id in node_id_set:                 # For every node ID in the list, from the first to the last:
            self.evaluate_new_node_status(node_id)  # evaluates the next status of "node_id" related perceptron




### Class for a basic tic tac toe game, including game rules and basic defense
##############################################################################
class myTris:

    #ATTRIBUTES(myTris):
    ####################
    perceptrons_network = None                  # Name of the perceptron network associated with the basic game
    max_number_of_perceptrons = None            # Max number of perceptrons allowed to be initialized in the network
    computer_victory_node_id = None             # This is the ID of the perceptron that becomes active when the computer wins
    list_of_computer_victory_node_ids = None    # This is the list of perceptrons that check if the computer has won
    human_victory_node_id = None                # This is the ID of the perceptron that becomes active when the user wins
    list_of_human_victory_node_ids = None       # This is the list of perceptrons that check if the user has won
    one_step_winning_node_id = None             # This is the ID of the perceptron that becomes active when the computer can win in one move
    list_of_node_ids_for_winning = None         # This is the list of perceptrons that check if the computer can win in one step
    activated_defense_node_id = None            # This is the ID of the perceptron that becomes active when the user can win in one move
    list_of_node_ids_for_defense = None         # This is the list of perceptrons that check if the user can win in one step
    list_of_node_ids_for_attack_random = None   # This is the list of perceptrons that allow the computer for a random move
    tie_node_id = None                          # This is the ID of the perceptron that becomes active when it is tie (full board)
    list_of_full_board_node_ids = None          # This is the list of perceptrons that check if the board is full (it is tie)

    #METHODS(myTris):
    #################

    #####################################################################################################
    ### The myTris constructor defines the parameters of the network, the network itself and the weighted 
    ### links to embody basic rules and defense.
    #####################################################################################################
    def __init__(self,starting_status = [EMPTY for i in range(9)],verbose = True): # the starting status of the board is EMPTY for every cell
        
        self.max_number_of_perceptrons = 300    # max number of perceptrons that can be initialized on the network
        
        net = myPerceptronNetwork("Main perceptron network",self.max_number_of_perceptrons) # Initialize the perceptron network
        if verbose: print("Created a network of",self.max_number_of_perceptrons,"available perceptrons (the first 9 are the board game)")
        
        ### The following is the basic training of the perceptron network.
        ### It is based on initializing nodes and creating weighted connections between them.
        ### Basically, it is a way to program the perceptron network for the basic context of the game.
        
        # INITIALIZE THE GAME BOARD (9 nodes):
        ####################################
        for i in range(9):
            node_id = net.new_node()                                                # Initialize a new node
            net.new_link(from_node_id = node_id,to_node_id = node_id, weight = 2.0) # Establish full feedback for every node of the board
        for i in range(9):                                      # for every node of the board (cell)
            net.perceptron_nodes[i].status = starting_status[i] # set the starting status of the perceptrons
        if verbose: print("Initialised board game:",net.network_dimension,"nodes (total).")

        # NETWORK TO IDENTIFY COMPUTER VICTORY OOO OR CIRCLE-CIRCLE-CIRCLE:
        ###################################################################
        self.computer_victory_node_id = net.new_node()      # Initialize the main node that is active iff there is a OOO in the board
        self.list_of_computer_victory_node_ids = list()     # Initialize the list of the perceptron ID for detecting the OOO
        for i in range(3):                                  # Loop for covering the rows of the board                       
            node_id = net.new_node()                                                            # Initialize a new node
            idx_list = [0+3*i,1+3*i,2+3*i]                                                      # creates the IDs list of the row
            net.node_inputs(to_node_id = node_id, input_list = [(k,1/3) for k in idx_list])     # detect OOO in rows
            self.list_of_computer_victory_node_ids.append(node_id)                              # add the node ID to the list
        for i in range(3):                                  # Loop for covering the columns of the board                       
            node_id = net.new_node()                                                            # Initialize a new node
            idx_list = [i,(i+3),(i+6)]                                                          # creates the IDs list of the column
            net.node_inputs(to_node_id = node_id, input_list = [(k,1/3) for k in idx_list])     # detect OOO in columns
            self.list_of_computer_victory_node_ids.append(node_id)                              # add the node ID to the list
        node_id = net.new_node()                                                        # Initialize a new node
        net.node_inputs(to_node_id = node_id, input_list = [(0,1/3),(4,1/3),(8,1/3)])   # detect OOO in the main Diagonal
        self.list_of_computer_victory_node_ids.append(node_id)                          # add the node ID to the list
        node_id = net.new_node()                                                        # Initialize a new node
        net.node_inputs(to_node_id = node_id, input_list = [(2,1/3),(4,1/3),(6,1/3)])   # detect OOO in the anti Diagonal
        self.list_of_computer_victory_node_ids.append(node_id)                          # add the node ID to the list
        # Set all node IDs in the list as inputs of the main node already defined at the beginning:
        net.node_inputs(to_node_id = self.computer_victory_node_id, input_list = [(idx,1) for idx in self.list_of_computer_victory_node_ids])
        if verbose: print("Initialised perceptron network for defining computer victory:",net.network_dimension,"nodes (total).")

        # NETWORK FOR BASIC DEFENSE STRATEGY AGAINST XX_ X_X _XX:
        #########################################################
        self.activated_defense_node_id = net.new_node()     # Initialize the main node that is active iff there is a XX_ X_X _XX in the board
        self.list_of_node_ids_for_defense = list()          # Initialize the list of the perceptron ID for detecting the XX_ X_X _XX
        for i in range(3):
            for j in range(3):                              # Loop for covering the rows of the board
                node_id = net.new_node()                    # Initialize a new node
                idx_list = [0+3*i,1+3*i,2+3*i]              # creates the IDs list of the row
                idx_list.remove(3*i+j)                      # remove just one cell
                net.node_inputs(to_node_id = node_id, input_list = [(k,-1/2) for k in idx_list])    # detect the attack on the row
                net.node_inputs(to_node_id = 3*i+j, input_list = [(node_id,1)])                     # push the removed cell
                self.list_of_node_ids_for_defense.append(node_id)                                   # add the node ID to the list
        for i in range(3):
            for j in range(3):                              # Loop for covering the columns of the board
                node_id = net.new_node()                    # Initialize a new node
                idx_list = [i,(i+3),(i+6)]                  # creates the IDs list of the column
                idx_list.remove(i+3*j)                      # remove just one cell
                net.node_inputs(to_node_id = node_id, input_list = [(k,-1/2) for k in idx_list])    # detect the attack on the column
                net.node_inputs(to_node_id = i+3*j, input_list = [(node_id,1)])                     # push the removed cell
                self.list_of_node_ids_for_defense.append(node_id)                                   # add the node ID to the list
        for j in range(3):                              # Loop for covering the main diagonal of the board
            node_id = net.new_node()                    # Initialize a new node
            idx_list = [0,4,8]                          # creates the IDs list of the main diagonal
            r = idx_list[j]                             # get the j-th cell ID of the list
            idx_list.remove(r)                          # remove just one cell
            net.node_inputs(to_node_id = node_id, input_list = [(k,-1/2) for k in idx_list])        # detect the attack on the main diagonal
            net.node_inputs(to_node_id = r, input_list = [(node_id,1)])                             # push the removed cell
            self.list_of_node_ids_for_defense.append(node_id)                                       # add the node ID to the list
        for j in range(3):                              # Loop for covering the anti-diagonal of the board
            node_id = net.new_node()                    # Initialize a new node
            idx_list = [2,4,6]                          # creates the IDs list of the anti-diagonal
            r = idx_list[j]                             # get the j-th cell ID of the list
            idx_list.remove(r)                          # remove just one cell
            net.node_inputs(to_node_id = node_id, input_list = [(k,-1/2) for k in idx_list])        # detect the attack on the anti-diagonal
            net.node_inputs(to_node_id = r, input_list = [(node_id,1)])                             # push the removed cell
            self.list_of_node_ids_for_defense.append(node_id)                                       # add the node ID to the list
        # Set all node IDs in the list as inputs of the main node already defined at the beginning:
        net.node_inputs(to_node_id = self.activated_defense_node_id, input_list = [(idx,1) for idx in self.list_of_node_ids_for_defense])
        if verbose: print("Initialised basic defensive strategy:",net.network_dimension,"nodes (total).")

        # NETWORK FOR ONE STEP WINNING BASIC STRATEGY OO_ O_O _OO:
        ##########################################################
        self.one_step_winning_node_id = net.new_node()      # Initialize the main node that is active iff there is a OO_ O_O _OO in the board
        self.list_of_node_ids_for_winning = list()          # Initialize the list of the perceptron ID for detecting the OO_ O_O _OO
        for i in range(3):
            for j in range(3):                              # Loop for covering the rows of the board
                node_id = net.new_node()                    # Initialize a new node
                idx_list = [0+3*i,1+3*i,2+3*i]              # creates the IDs list of the row
                idx_list.remove(3*i+j)                      # remove just one cell
                net.node_inputs(to_node_id = node_id, input_list = [(k,1/2) for k in idx_list])     # detect the winning pos on the row
                net.node_inputs(to_node_id = 3*i+j, input_list = [(node_id,1)])                     # push the removed cell
                self.list_of_node_ids_for_winning.append(node_id)                                   # add the node ID to the list
        for i in range(3):
            for j in range(3):                              # Loop for covering the columns of the board
                node_id = net.new_node()                    # Initialize a new node
                idx_list = [i,(i+3),(i+6)]                  # creates the IDs list of the column
                idx_list.remove(i+3*j)                      # remove just one cell
                net.node_inputs(to_node_id = node_id, input_list = [(k,1/2) for k in idx_list])     # detect the winning pos on the column
                net.node_inputs(to_node_id = i+3*j, input_list = [(node_id,1)])                     # push the removed cell
                self.list_of_node_ids_for_winning.append(node_id)                                   # add the node ID to the list
        for j in range(3):                              # Loop for covering the main diagonal of the board
            node_id = net.new_node()                    # Initialize a new node
            idx_list = [0,4,8]                          # creates the IDs list of the main diagonal
            r = idx_list[j]                             # get the j-th cell ID of the list
            idx_list.remove(r)                          # remove just one cell
            net.node_inputs(to_node_id = node_id, input_list = [(k,1/2) for k in idx_list])         # detect the winning pos on the main diagonal
            net.node_inputs(to_node_id = r, input_list = [(node_id,1)])                             # push the removed cell
            self.list_of_node_ids_for_winning.append(node_id)                                       # add the node ID to the list
        for j in range(3):                              # Loop for covering the anti-diagonal of the board
            node_id = net.new_node()                    # Initialize a new node
            idx_list = [2,4,6]                          # creates the IDs list of the anti-diagonal
            r = idx_list[j]                             # get the j-th cell ID of the list
            idx_list.remove(r)                          # remove just one cell
            net.node_inputs(to_node_id = node_id, input_list = [(k,1/2) for k in idx_list])         # detect the winning pos on the anti-diagonal
            net.node_inputs(to_node_id = r, input_list = [(node_id,1)])                             # push the removed cell
            self.list_of_node_ids_for_winning.append(node_id)                                       # add the node ID to the list
        # Set all node IDs in the list as inputs of the main node already defined at the beginning:
        net.node_inputs(to_node_id = self.one_step_winning_node_id, input_list = [(idx,1) for idx in self.list_of_node_ids_for_winning])
        if verbose: print("Initialised basic winning strategy:",net.network_dimension,"nodes (total).")

        # NETWORK TO IDENTIFY USER VICTORY XXX OR STAR-STAR-STAR:
        #########################################################
        self.human_victory_node_id = net.new_node()         # Initialize the main node that is active iff there is a XXX in the board
        self.list_of_human_victory_node_ids = list()        # Initialize the list of the perceptron ID for detecting the XXX
        for i in range(3):                                  # Loop for covering the rows of the board                       
            node_id = net.new_node()                                                            # Initialize a new node
            idx_list = [0+3*i,1+3*i,2+3*i]                                                      # creates the IDs list of the row
            net.node_inputs(to_node_id = node_id, input_list = [(k,-1/3) for k in idx_list])    # detect XXX in rows
            self.list_of_human_victory_node_ids.append(node_id)                                 # add the node ID to the list
        for i in range(3):                                  # Loop for covering the columns of the board                       
            node_id = net.new_node()                                                            # Initialize a new node
            idx_list = [i,(i+3),(i+6)]                                                          # creates the IDs list of the column
            net.node_inputs(to_node_id = node_id, input_list = [(k,-1/3) for k in idx_list])    # detect XXX in columns
            self.list_of_human_victory_node_ids.append(node_id)                                 # add the node ID to the list
        node_id = net.new_node()                                                                # Initialize a new node
        net.node_inputs(to_node_id = node_id, input_list = [(0,-1/3),(4,-1/3),(8,-1/3)])        # detect XXX in the Main Diagonal
        self.list_of_human_victory_node_ids.append(node_id)                                     # add the node ID to the list
        node_id = net.new_node()                                                                # Initialize a new node
        net.node_inputs(to_node_id = node_id, input_list = [(2,-1/3),(4,-1/3),(6,-1/3)])        # detect XXX in the anti-diagonal
        self.list_of_human_victory_node_ids.append(node_id)                                     # add the node ID to the list
        # Set all node IDs in the list as inputs of the main node already defined at the beginning:
        net.node_inputs(to_node_id = self.human_victory_node_id, input_list = [(idx,1) for idx in self.list_of_human_victory_node_ids])
        if verbose: print("Initialised perceptron network for defining human victory:",net.network_dimension,"nodes (total).")

        # NETWORK TO MODEL A RANDOM ATTACK STRATEGY:
        ############################################
        self.list_of_node_ids_for_attack_random = list()    # Initialize the list of the perceptron IDs, one for every board cell
        for i in range(9):                                  # Loop: for every cell of the board
            node_id = net.new_node()                                                # Initialize a new node
            net.node_inputs(to_node_id = node_id, input_list = [(i,1),(None,1)])    # feedback on the cell i-th and a link-unrelated weight
            net.node_inputs(to_node_id = i, input_list = [(node_id,1)])             # The output of a new node is the input of a board-cell
            self.list_of_node_ids_for_attack_random.append(node_id)                 # add the node ID to the list
        if verbose: print("Initialised perceptron network for random attack strategy:",net.network_dimension,"nodes (total).")


        # NETWORK FOR DETECTING A TIE (FULL BOARD):
        ###########################################
        self.list_of_full_board_node_ids = list()   # Initialize the list of the perceptron IDs for evaluating the tie (full board)
        for i in range(9):   # create 9 nodes coming from every cell of the board with a link weighted -1
            node_id = net.new_node()                                        # Initialize a new node
            net.node_inputs(to_node_id = node_id, input_list = [(i,-1)])    # detect X (STAR) in the cell
            self.list_of_full_board_node_ids.append(node_id)                # add the node ID to the list
        for i in range(9):  # create 9 nodes coming from every cell of the board with a link weighted 1
            node_id = net.new_node()                                        # Initialize a new node
            net.node_inputs(to_node_id = node_id, input_list = [(i,1)])     # detect O (CIRCLE) in the cell
            self.list_of_full_board_node_ids.append(node_id)                # add the node ID to the list
        node_id_1 = net.new_node()                                          # Initialize a new node
        net.node_inputs(to_node_id = node_id_1, input_list = [(node_id-9-i,1/5) for i in range(9)])   # detect at least 5 STARS
        self.list_of_full_board_node_ids.append(node_id_1)                  # add the node ID to the list
        node_id_2 = net.new_node()                                          # Initialize a new node
        net.node_inputs(to_node_id = node_id_2, input_list = [(node_id-i,1/4) for i in range(9)])   # detect at least 4 CIRCLES
        self.list_of_full_board_node_ids.append(node_id_2)                  # add the node ID to the list
        TIE_WITH_HUMAN_PREVALENCE_NODE_ID = net.new_node()                  # Initialize the node that identify X (STAR) prevalence
        net.node_inputs(to_node_id = TIE_WITH_HUMAN_PREVALENCE_NODE_ID, input_list = [(node_id_1,1/2),(node_id_2,1/2)]) # detect X prevalence
        self.list_of_full_board_node_ids.append(TIE_WITH_HUMAN_PREVALENCE_NODE_ID)  # add the node ID to the list
        for i in range(9):  # create 9 nodes coming from every cell of the board with a link weighted 1
            node_id = net.new_node()                                        # Initialize a new node
            net.node_inputs(to_node_id = node_id, input_list = [(i,1)])     # detect O (CIRCLE) in the cell
            self.list_of_full_board_node_ids.append(node_id)                # add the node ID to the list
        for i in range(9):  # create 9 nodes coming from every cell of the board with a link weighted -1
            node_id = net.new_node()                                        # Initialize a new node
            net.node_inputs(to_node_id = node_id, input_list = [(i,-1)])    # detect X (STAR) in the cell
            self.list_of_full_board_node_ids.append(node_id)                # add the node ID to the list
        node_id_1 = net.new_node()                                          # Initialize a new node
        net.node_inputs(to_node_id = node_id_1, input_list = [(node_id-9-i,1/5) for i in range(9)]) # detect at least 5 CIRCLES
        self.list_of_full_board_node_ids.append(node_id_1)                  # add the node ID to the list
        node_id_2 = net.new_node()                                          # Initialize a new node 
        net.node_inputs(to_node_id = node_id_2, input_list = [(node_id-i,1/4) for i in range(9)])   # detect at least 4 STARS
        self.list_of_full_board_node_ids.append(node_id_2)                  # add the node ID to the list
        TIE_WITH_COMPUTER_PREVALENCE_NODE_ID = net.new_node()               # Initialize a new node
        net.node_inputs(to_node_id = TIE_WITH_COMPUTER_PREVALENCE_NODE_ID, input_list = [(node_id_1,1/2),(node_id_2,1/2)]) # detect O prevalence
        self.list_of_full_board_node_ids.append(TIE_WITH_COMPUTER_PREVALENCE_NODE_ID)   # add the node ID to the list
        self.tie_node_id = net.new_node()                                   # Initialize a new node
        # if the prevalence of X (5 X and 4 O) or the prevalence of O (5 O and 4 X) determines a tie:
        net.node_inputs(to_node_id = self.tie_node_id, input_list = [(TIE_WITH_HUMAN_PREVALENCE_NODE_ID,1),(TIE_WITH_COMPUTER_PREVALENCE_NODE_ID,1)])
        self.list_of_full_board_node_ids.append(self.tie_node_id)           # add the node ID to the list

        if verbose: print("Initialised perceptron network for full board (tie) detection:",net.network_dimension,"nodes (total).")
        
        self.perceptrons_network = net  # set the object attribute "perceptrons_network" to the contents of the processed local variable "net"

        if verbose: print("Basic initialization done: nr",net.network_dimension,"perceptrons out of",self.max_number_of_perceptrons)

    ##################################################################################### 
    ### myTris method "try_move" evaluates the next status of each cell
    ### in the game board, so the first 8 cells of the perceptron network. The order 
    ### of evaluation is random and the procedure ends at the first evaluated cell 
    ### that becomes active (change of state).
    #####################################################################################  
    def try_move(self):
        tris_board = [0,1,2,3,4,5,6,7,8]    # set the board cell (perceptron) ID list
        random.shuffle(tris_board)          # apply order randomization
        for cell in tris_board:             # for every cell in the board...
            if self.perceptrons_network.evaluate_new_node_status(cell) == "move_done":      # ... evaluates the next status and
                return "move_done"                                      # if the cell-status is changed, then return "move_done"
        return "no_move"                    # otherwise return "no_move" performed

    ###########################################################################################
    ### myTris method that reset the state of every node (perceptron) to EMPTY
    ### except the game board. It is very important to reset the behavior of the perceptrons
    ### during a single game.
    ###########################################################################################       
    def reset_all_but_the_board(self):
        # Consider all nodes except the initial 8 that make up the game board:
        for i in range(9,self.perceptrons_network.network_dimension):   
            self.perceptrons_network.perceptron_nodes[i].status = EMPTY # set them to EMPTY, so 0.0
            
    ########################################################################################### 
    ### The myTris method "respond" receives a board state as input and provides as output
    ### the following triple (based only on its basic training):
    ### 1. the evaluated situation of the game (e.g. computer_victory, tie, etc.)
    ### 2. the initial status of the board (so the input)
    ### 3. the software's response to the situation, that is the new status of the board
    ###########################################################################################   
    def respond(self,from_status = [EMPTY for i in range(9)]):  # the starting status by default is a board with all cells EMPTY
        # set the starting status of the board, cell by cell:
        for i in range(9):
            self.perceptrons_network.perceptron_nodes[i].status = from_status[i]
        # all perceptrons related to the computer's victory status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_computer_victory_node_ids)
        # If the following node is active, the game is over and the result is the computer's victory:
        if self.perceptrons_network.evaluate_new_node_status(self.computer_victory_node_id) == "activated_node": 
            return ("computer_victory",from_status,from_status)
        # all perceptrons related to user victory status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_human_victory_node_ids)
        # if the following node is active then the game is over and the result is user victory:
        if self.perceptrons_network.evaluate_new_node_status(self.human_victory_node_id) == "activated_node": 
            return ("human_victory",from_status,from_status)
        # all perceptrons related to full board status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_full_board_node_ids)
        # if the following node is active then the game is over and the result is tie (the board is full):
        if self.perceptrons_network.evaluate_new_node_status(self.tie_node_id) == "activated_node": 
            return ("tie",from_status,from_status)
        # all perceptrons related to one step to win for computer are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_for_winning)
        # if the following node is active then the computer makes the last move:
        if self.perceptrons_network.evaluate_new_node_status(self.one_step_winning_node_id) == "activated_node":
            if self.try_move() == "move_done":
                # if the move has been done then the game is over with computer victory
                to_status = list()  # collect the resulting status of the board, the 9 values of the 9 cells
                for i in range(9):
                    to_status.append(self.perceptrons_network.perceptron_nodes[i].status)
                return ("computer_victory",from_status,to_status)
        # all perceptrons related to one step to win for user are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_for_defense)
        # if the following node is active then the computer makes a defensive move:
        if self.perceptrons_network.evaluate_new_node_status(self.activated_defense_node_id) == "activated_node":
            if self.try_move() == "move_done":
                # if the move has been done then the game continues
                to_status = list()
                for i in range(9):
                    to_status.append(self.perceptrons_network.perceptron_nodes[i].status)
                return ("basic_defense",from_status,to_status)
        # all the perceptrons related to one step to a random attack are evaluated:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_for_attack_random)
        # the computer randomly tries to activate one of the cells of the board:
        if self.try_move() == "move_done":
            # if a move has been done then the game continues
            to_status = list()
            for i in range(9):
                to_status.append(self.perceptrons_network.perceptron_nodes[i].status)
            return ("random_attack",from_status,to_status)
        # if here then there is no response from the software to the board-status of the input (maybe an error occurred)
        return("unable_to_respond",from_status,from_status)

    #################################################################################
    ### The myTris method "show" draws the tris game on the screen based on the board
    ### (first 9 perceptrons of the network)
    #################################################################################
    def show(self):

        def convert(status):    # convert: CIRCLE, so 1.0 to O, STAR, so -1,0 to X and EMPTY, so 0.0 to _
            if status == CIRCLE:
                return "O"
            elif status == STAR:
                return "X"
            elif status == EMPTY:
                return "_"
            else:
                print("Error 1 from class myTris: bad status value [",status,"]")
                quit()

        # convert all the cells of the board and print:
        v = [convert(self.perceptrons_network.perceptron_nodes[i].status) for i in range(9)]
        print("|",v[0],v[1],v[2],"| 012")
        print("|",v[3],v[4],v[5],"| 345")
        print("|",v[6],v[7],v[8],"| 678")




### Class for a tris game that includes lessons-learnt from matches
###################################################################
class myTrainedTris(myTris):    # This class inherits properties from myTris one.

    #ATTRIBUTES(myTrainedTris):
    ###########################
    # ID of the perceptron which is active when you can use a lesson learned on how to win for context:
    recognised_lessons_learnt_win_node_id = None
    # list of the IDs of the nodes which are used to evaluate whether the context corresponds to a winning strategy:
    list_of_node_ids_from_lessons_learnt_win = None
    # ID of the perceptron which is active when you can use a lesson learned on how to get the draw for the context:
    recognised_lessons_learnt_tie_node_id = None
    # list of the IDs of the nodes which serve to evaluate whether the context corresponds to a strategy to obtain a tie:
    list_of_node_ids_from_lessons_learnt_tie = None
    # ID of the perceptron which is active when you can use a lesson learned on how not to lose for context:
    recognised_lessons_learnt_not_loosing_node_id = None
    # list of the IDs of the nodes which are used to evaluate whether the context corresponds to a non-losing strategy:
    list_of_node_ids_from_lessons_learnt_not_loosing = None
    
    match = None                # a match is a list of 10 elements: the first one is the player that begins
                                # (CIRCLE or STAR), the remaining nine are the IDs of the cells covered during
                                # the game session. E.g. [CIRCLE,0,4,3,5,7,8,1,2,6]
    match_move_counter = None   # This is the counter from 0 to 9 for filling the previous list during the match.

    #METHODS(myTrainedTris):
    ########################

    #####################################################################################
    ### The myTrainedTris constructor calls the inherited constructor from the myTris 
    ### class, then adds a training for the perceptron network based on the information  
    ### in 3 external text files:
    ### - mytris.lessonslearnt_not_loose.txt
    ### - mytris.lessonslearnt_tie.txt
    ### - mytris.lessonslearnt_win.txt
    ### built by the same software during the matches (experience).
    #####################################################################################
    def __init__(self,starting_status = [EMPTY for i in range(9)],verbose = True):
        
        def load_from_file(verbose,my_kb_file_name):    # load the file whose name is contained in the variable "my_kb_file_name"
                                                        # structure: list of ( board description , next move to do)
            lesson_learnt_kb = list()
            if os.path.exists(my_kb_file_name):
                if verbose: print("Loading lessons-learnt knowledge base from file [",my_kb_file_name,"]:")
                if verbose: print("Records: ",end="")
                with open(my_kb_file_name, 'rt') as my_file_handler:
                    counter = 0
                    val_list = list()
                    while True:
                        for i in range(9):
                            r = my_file_handler.readline().strip()
                            if not r:
                                break 
                            val_list.append(float(r))
                        r = my_file_handler.readline().strip()
                        if not r:
                            break 
                        lesson_learnt_kb.append((val_list,int(r)))  
                        val_list = list()
                        counter += 1
                        if verbose: print(counter,",",end="")
                    my_file_handler.close()
                if verbose: print("done.")
                return lesson_learnt_kb
            else:
                if verbose: print("No lessons-learnt knowledge base file [",my_kb_file_name,"] found.")
                return []
            
        super().__init__(starting_status,verbose)   # invoke the inherited constructor from myTris class
        self.match = [None for i in range(10)]      # set the starting values of match list to None
        self.match_move_counter = 0                 # set the related counter to zero
        
        if verbose:
            print()
            print("Using lessons learnt for perceptron network training...")
            print()
        
        # BUILD AND TRAIN THE NETWORK-PART FOR LESSONS LEARNT ABOUT WINNING (ATTACKING STRATEGY):
        #########################################################################################
        # Load information from the knowledge base file named mytris.lessonslearnt_win.txt:
        LESSON_LEARNT_KNOWLEDGE_BASE = load_from_file(verbose,"mytris.lessonslearnt_win.txt")
        l = len(LESSON_LEARNT_KNOWLEDGE_BASE)
        self.list_of_node_ids_from_lessons_learnt_win = []  # Initialize the list of structured info
        # if the file is empty don't do anything
        if l == 0:
            if verbose: print("Loaded no rules from lessons learnt knowledge base (it is empty).")
        else:
            for (w,k) in LESSON_LEARNT_KNOWLEDGE_BASE:          # for each pair in the list:
                node_id = self.perceptrons_network.new_node()   # Initialize a new node
                # apply the information to identify whether the card context is recognized:
                self.perceptrons_network.node_inputs(to_node_id = node_id, input_list = [(i,w[i]) for i in range(9)] )
                # if the context matches, set the k-th cell as next move:
                self.perceptrons_network.node_inputs(to_node_id = k, input_list = [(node_id,1)]) 
                self.list_of_node_ids_from_lessons_learnt_win.append(node_id)                   # add the new node ID to the list
            self.recognised_lessons_learnt_win_node_id = self.perceptrons_network.new_node()    # Initialize a new node
            # set all previous nodes as inputs for this one which will be active only when context is good:
            self.perceptrons_network.node_inputs(to_node_id = self.recognised_lessons_learnt_win_node_id, 
            input_list = [(idx,1) for idx in self.list_of_node_ids_from_lessons_learnt_win])
            if verbose: print("Imported",l,"rules from lessons learnt knowledge base for winning.")

        if verbose: print()
        
        # BUILD AND TRAIN THE NETWORK-PART FOR LESSONS LEARNT ABOUT GETTING TIE:
        ########################################################################
        # Load information from the knowledge base file named mytris.lessonslearnt_tie.txt:
        LESSON_LEARNT_KNOWLEDGE_BASE = load_from_file(verbose,"mytris.lessonslearnt_tie.txt")
        l = len(LESSON_LEARNT_KNOWLEDGE_BASE)
        self.list_of_node_ids_from_lessons_learnt_tie = []  # Initialize the list of structured info
        # if the file is empty don't do anything
        if l == 0:
            if verbose: print("Loaded no rules from lessons learnt knowledge base (it is empty).")
        else:
            for (w,k) in LESSON_LEARNT_KNOWLEDGE_BASE:          # for each pair in the list:
                node_id = self.perceptrons_network.new_node()   # Initialize a new node
                # apply the information to identify whether the card context is recognized:
                self.perceptrons_network.node_inputs(to_node_id = node_id, input_list = [(i,w[i]) for i in range(9)] )
                # if the context matches, set the k-th cell as next move:
                self.perceptrons_network.node_inputs(to_node_id = k, input_list = [(node_id,1)]) 
                self.list_of_node_ids_from_lessons_learnt_tie.append(node_id)   # add the new node ID to the list
            self.recognised_lessons_learnt_tie_node_id = self.perceptrons_network.new_node()    # Initialize a new node
            # set all previous nodes as inputs for this one which will be active only when context is good:
            self.perceptrons_network.node_inputs(to_node_id = self.recognised_lessons_learnt_tie_node_id, 
            input_list = [(idx,1) for idx in self.list_of_node_ids_from_lessons_learnt_tie])
            if verbose: print("Imported",l,"rules from lessons learnt knowledge base for tie.")  
            
        if verbose: print()

        # BUILD AND TRAIN THE NETWORK-PART FOR LESSONS LEARNT ABOUT NOT LOOSING (DEFENSIVE STRATEGY):
        #############################################################################################
        # Load information from the knowledge base file named mytris.lessonslearnt_not_loose.txt:
        LESSON_LEARNT_KNOWLEDGE_BASE = load_from_file(verbose,"mytris.lessonslearnt_not_loose.txt")
        l = len(LESSON_LEARNT_KNOWLEDGE_BASE)
        self.list_of_node_ids_from_lessons_learnt_not_loosing = []  # Initialize the list of structured info
        # if the file is empty don't do anything
        if l == 0:
            if verbose: print("Loaded no rules from lessons learnt knowledge base (it is empty).")
        else:
            for (w,k) in LESSON_LEARNT_KNOWLEDGE_BASE:          # for each pair in the list:
                node_id = self.perceptrons_network.new_node()   # Initialize a new node
                # apply the information to identify whether the card context is recognized:
                self.perceptrons_network.node_inputs(to_node_id = node_id, input_list = [(i,w[i]) for i in range(9)] )
                # if the context matches, set the k-th cell as next move:
                self.perceptrons_network.node_inputs(to_node_id = k, input_list = [(node_id,1)]) 
                self.list_of_node_ids_from_lessons_learnt_not_loosing.append(node_id)   # add the new node ID to the list
            self.recognised_lessons_learnt_not_loosing_node_id = self.perceptrons_network.new_node()    # Initialize a new node
            # set all previous nodes as inputs for this one which will be active only when context is good:
            self.perceptrons_network.node_inputs(to_node_id = self.recognised_lessons_learnt_not_loosing_node_id, 
            input_list = [(idx,1) for idx in self.list_of_node_ids_from_lessons_learnt_not_loosing])
            if verbose: print("Imported",l,"rules from lessons learnt knowledge base for not loosing.") 
            
        if verbose:
            print()
            print("Total: used nr",self.perceptrons_network.network_dimension,"perceptrons out of",self.max_number_of_perceptrons)

    ##################################################################################################################
    ### The myTrainedTris method "check" evaluates whether there is a win or a draw between computer/user on the board
    ##################################################################################################################
    def check(self,verbose = True):
        # all perceptrons related to computer victory status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_computer_victory_node_ids)
        # if the following node is active then the game is over and the result is computer victory:
        if self.perceptrons_network.evaluate_new_node_status(self.computer_victory_node_id) == "activated_node":
            if verbose: print("I have won!")
            return "computer_victory"
        # all perceptrons related to user victory status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_human_victory_node_ids)
        # if the following node is active then the game is over and the result is user victory:
        if self.perceptrons_network.evaluate_new_node_status(self.human_victory_node_id) == "activated_node":
            if verbose: print("Great, You have won!")
            return "human_victory"
        # all perceptrons related to tie status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_full_board_node_ids)
        # if the following node is active then the game is over and the result is tie:
        if self.perceptrons_network.evaluate_new_node_status(self.tie_node_id) == "activated_node": 
            if verbose: print("It's a tie!")
            return "tie"

    ##############################################################################################################
    ### The myTrainedTris method "get_computer_move" evaluates the computer's next move based on the current state
    ### of the board, using basic knowledge and experience (lessons learnt).
    ##############################################################################################################
    def get_computer_move(self,verbose = True):
        # all perceptrons related to full board status are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_full_board_node_ids)
        # if the following node is active then the game is over and the result is impossible to make a move:
        if self.perceptrons_network.evaluate_new_node_status(self.tie_node_id) == "activated_node":
            if verbose: print("Board if full! I cannot move...")
            return "no_possible_move"
        # all perceptrons related to elementary winning strategy are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_for_winning)
        # if the following node is active then try to make a move and win:
        if self.perceptrons_network.evaluate_new_node_status(self.one_step_winning_node_id) == "activated_node":
            if self.try_move() == "move_done":
                if verbose: print("With the next move I win...")
                return "one_step_winning"
        # all perceptrons related to elementary defensive strategy are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_for_defense)
        # if the following node is active then try to make a move and defend:
        if self.perceptrons_network.evaluate_new_node_status(self.activated_defense_node_id) == "activated_node":
            if self.try_move() == "move_done":
                if verbose: print("My next move will be a basic defense...")
                return "basic_defense"
        # if info from experience are available on related files:
        if self.list_of_node_ids_from_lessons_learnt_not_loosing != []:
            # all perceptrons related to lessons-learnt for not loosing strategies are evaluated in sequence:
            self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_from_lessons_learnt_not_loosing)
            # if the following node is active then try to make a move and defend:
            if self.perceptrons_network.evaluate_new_node_status(self.recognised_lessons_learnt_not_loosing_node_id) == "activated_node":
                if self.try_move() == "move_done" :
                    if verbose: print("With my next move I will defend based on what I learned from the games...")
                    return "learnt_defense"
        # if info from experience are available on related files:
        if self.list_of_node_ids_from_lessons_learnt_win != []:
            # all perceptrons related to lessons-learnt for winning strategies are evaluated in sequence:
            self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_from_lessons_learnt_win)
            # if the following node is active then try to make a move and attack:
            if self.perceptrons_network.evaluate_new_node_status(self.recognised_lessons_learnt_win_node_id) == "activated_node":
                if self.try_move() == "move_done":
                    if verbose: print("My next move will be based on the lessons learned from a victory...")
                    return "lessons_learnt_winning_attack"
        # if info from experience are available on related files:
        if self.list_of_node_ids_from_lessons_learnt_tie != []:
            # all perceptrons related to lessons-learnt for tie strategies are evaluated in sequence:
            self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_from_lessons_learnt_tie)
            # if the following node is active then try to move and defend:
            if self.perceptrons_network.evaluate_new_node_status(self.recognised_lessons_learnt_tie_node_id) == "activated_node":
                if self.try_move() == "move_done":
                    if verbose: print("My next move will be based on a lesson learned from a draw...")
                    return "lessons_learnt_tie_attack"
        # if nothing worked then apply a random strategy...
        # all perceptrons related to a random strategy are evaluated in sequence:
        self.perceptrons_network.evaluate_new_status_for_all_nodes_sequentially(self.list_of_node_ids_for_attack_random)
        # try to randomly make a move
        if self.try_move() == "move_done":
            if verbose: print("My next move is random, I can't do better in this situation...")
            return "random_attack"
        # The following part should never be reachable, it's just a precaution...
        if verbose: print("I don't know what to do, I'm so sorry!!!")
        return "unable_to_respond"

    ####################################################################
    ### The myTrainedTris method "get_user_move" gets the user's next 
    ### move. The user can enter a number between 0 and 8 (inclusive) 
    ### that defines where he or she would like to place the STAR.
    ####################################################################
    def get_user_move(self,verbose = True):
        while True:                                                 # repeat the request to the user until the number is acceptable
            cs = input("Please select a cell between 0 and 8: ")
            try:
                c = int(cs)
            except ValueError:
                print("Warning! only numeric values between 0 and 8!")          # integers only allowed
                continue            
            if  c < 0 or c > 8:                                                 # integers only between 0 and 9 (inclusive)
                print("Sorry, number",c,"is wrong... choose another cell id!")
                continue
            if self.perceptrons_network.perceptron_nodes[c].status == EMPTY:    # a STAR can be set in a cell iff the cell is EMPTY
                self.perceptrons_network.perceptron_nodes[c].status = STAR
                return "move_done"
            else:
                print("Sorry, cell",c,"is busy... choose another cell id!")

    ####################################################################
    ### The myTrainedTris method "play" manages the game between user
    ### and computer/software.
    ####################################################################
    def play(self):
        start = input("Would you like to be the first to make a move? (Yes = y, otherwise No) ")    # define who is the first to play
        if start == "y" or start == "Y":
            turn = "user"
            self.match[self.match_move_counter] = STAR      # Initialize the 0 position (first player) of the match list to user (STAR)
        else:
            turn = "computer"
            self.match[self.match_move_counter] = CIRCLE    # Initialize the 0 position (first player) of the match list to computer (CIRCLE)
        self.match_move_counter += 1                        # Increase the counter of the match list
        while True:                                         # Cycle for the game:
            #self.perceptrons_network.reset_all_but_the_board()  # reset all perceptron network but board to avoid bad memory
            self.reset_all_but_the_board()  # reset all perceptron network but board to avoid bad memory
            if turn == "user":
                self.get_user_move()        # ask the user for a move
                turn = "computer"
            elif turn == "computer": 
                self.get_computer_move()    # ask the software for a move
                turn = "user"
            else:
                print("Error 1 from class myTrainedTris: bad turn [",turn,"]")
                return "end"
            # verify is a new move changed the board:
            for i in range(9):
                if self.perceptrons_network.perceptron_nodes[i].status != EMPTY:    # a move is a cell that does not contain EMPTY
                    move_found = False                                          # a move is a cell whose ID is not in the list of the match
                    for j in range(1,self.match_move_counter):
                        if i == self.match[j]:
                            move_found = True
                            break
                    if not move_found:                              # if there is a new move that changed the board and
                                                                    # it is not in the list of the match:                  
                        self.match[self.match_move_counter] = i     # add the new move the the list of the match
                        self.match_move_counter += 1                # increase the counter of the list of the match
                        break
            self.show()         # show the board to the user that is playing
            
            r = self.check()    # check the board status:
            if (r == "computer_victory" or r == "human_victory" or r == "tie"):     # if it is a game over:
                # ask the user if he wants to allow the software to learn the lesson from the match:
                ans = input("Do you want me to learn the basic scheme of this match? ( Y = yes, No otherwise ) ")
                if (ans == "y" or ans == "Y"):
                    # for a win, the software can learn lessons to win and lessons to not lose:
                    if r == "computer_victory" or r == "human_victory":
                        print()
                        print("I'm learning this match as a winning scheme:")
                        counter = 0
                        for i in range(1,10):           # counts values ​​other than None in the list of the match
                            if self.match[i] != None:   # this number defines who wins at the end of the game.
                                counter += 1
                        if (-1)**counter == 1:          # if count is even the second who started playing has won
                            self.match[0] = STAR
                        else:
                            self.match[0] = CIRCLE      # if count is odd the first who started playing has won
                        game_learning_for_winning = myGameLearning()
                        game_learning_for_winning.analyze_my_match(self.match,"win")    # activate the learning process by class myGameLearning
                        print()
                        print("I'm learning this match as a non loosing scheme:")
                        if (-1)**counter == 1:
                            self.match[0] = CIRCLE      # if count is even the first who started playing has lost
                        else:
                            self.match[0] = STAR        # if count is odd the second who started playing has lost
                        game_learning_for_not_loosing = myGameLearning()
                        game_learning_for_not_loosing.analyze_my_match(self.match,"loose")  # activate the learning process by class myGameLearning
                    # for a tie, the software can learn lessons for defending:
                    elif r == "tie":
                        print()
                        print("I'm learning this game as a draw pattern:")
                        game_learning_for_tie = myGameLearning()
                        game_learning_for_tie.analyze_my_match(self.match,"tie")    # activate the learning process by class myGameLearning
                # if the user does not authorize the learning procedure everything ends:
                print("End.")
                return "end"




### Class for the dynamic learning procedure of the perceptron network
######################################################################
class myGameLearning:

    #ATTRIBUTES(myGameLearning):
    ############################
    lessons_learnt_for_winning = None       # list of the lessons learnt as a winning strategy
    lessons_learnt_for_tie = None           # list of the lessons learnt as a tie strategy
    lessons_learnt_for_not_loosing = None   # list of the lessons learnt as a non loosing strategy

    #METHODS(myGameLearning):
    #########################

    ########################################################################
    ### The myGameLearning method "analyze_my_match" works on game history
    ### to define lessons learned that will be stored in 3 text files.
    ########################################################################
    def analyze_my_match(self,match,match_status):

        ####################################################################################
        ### The local procedure "analyze_single_match_if_win_or_tie" works on the
        ### list of the match only when the match ended with a winner or a draw.
        ####################################################################################
        def analyze_single_match_if_win_or_tie(match):
            l = len(match)
            if l != 10:     # the length of the list of the match must be 10, otherwise error:
                print("Error 1 from class myGameLearning: bad match-list [",match,"]")
                quit()
            player = match[0]       # the first position of the list of the match is the first player
            lessons = list()        # initialize the list of the lessons learnt
            # set some parameters to process the lessons learned:
            if player == CIRCLE:
                idx_start = 1
                myexp = 1
            elif player == STAR:
                idx_start = 2
                myexp = 0
            else:
                print("Error 2 from class myGameLearning: bad player...[",player,"]")
                quit()
            # Evaluate the final status of the board after playing accordingly with the list of the match:
            final_status = [EMPTY for i in range(9)]
            for i in range(1,10):
                if match[i] != None:
                    final_status[match[i]] = (-1)**(i-myexp)
            tris_check = myTris(verbose = False)            # use the class myTris to work on the board and
            (reason,_,_) = tris_check.respond(final_status) # evaluate the final status of the board after the match
            if reason != "computer_victory" and reason != "tie":
                print("Error 3 from class myGameLearning: bad match final reason...[",match,reason,"]")
                quit()
            # if here the response from myTris class is that the board (at the end of match) describes a victory or a tie
            for idx in range(idx_start,9,2):
                # analyze every move of the game and get evolution patterns to remember (something like rules or strategies):
                from_status = [EMPTY for i in range(9)]
                for i in range(1,idx):
                    if match[i] != None:
                        from_status[match[i]] = (-1)**(i-myexp)     # define the starting status
                to_status = [EMPTY for i in range(9)]
                for i in range(1,idx+1):
                    if match[i] != None:
                        to_status[match[i]] = (-1)**(i-myexp)       # define the target status
                tris = myTris(verbose = False)                      # use the class myTris to work on the board and
                (reason,_,_) = tris.respond(from_status)            # evaluate the final status of the board after the match
                if reason == "computer_victory" or reason == "tie":
                    # if here the situation of the board is a game over so nothing to learn (the last moves cannot be avoided)
                    continue
                if reason == "unable_to_respond" or reason == "random_attack":
                    # If the myTris class doesn't know how to make a move, it's okay to get a strategy by remembering the game move;
                    # create the strategy: match a corresponding board pattern and the right move:
                    weights = [0 for i in range(9)]                 # Initialize the list of weights
                    acc = 0                                         # Initialize the counter of the non EMPTY cells that didn't change
                    for j in range(9):
                        if from_status[j] == to_status[j]:
                            if from_status[j] != EMPTY:
                                acc += 1
                    destination_node_id = -1                        # Initialize the destination node ID with an impossible value
                                                                    # (IDs can be 0 or positive, not negative)
                    for j in range(9):
                        if from_status[j] == CIRCLE and to_status[j] == CIRCLE: # no status change for CIRCLE: 
                            weights[j] = 1/acc                                  # associate a positive weight 1/acc
                        elif from_status[j] == STAR and to_status[j] == STAR:   # no status change for STAR: 
                            weights[j] = -1/acc                                 # associate a negative weight -1/acc
                        elif from_status[j] == EMPTY and to_status[j] != EMPTY: # status change from EMPTY to non EMPTY: 
                            destination_node_id = j                             # define the destination node ID (the cell for the good move)
                        elif from_status[j] == EMPTY and to_status[j] == EMPTY: # no status change for EMPTY: 
                            weights[j] = 0                                      # associate a weight 0
                        else:
                            print("Error 4 from class myGameLearning: bad tris configuration...[",j,from_status,to_status,"]")
                            quit()
                    if destination_node_id < 0:     # if the destination node ID is not good an error occurred:
                        print("Error 5 from class myGameLearning: bad destination node id...[",destination_node_id,"]")
                        quit()
                    lessons.append((weights,destination_node_id))   # add the new lesson learnt (board pattern,move) to the list
            return lessons                                          # return the list of all the lessons learnt

        ####################################################################################
        ### The local procedure "analyze_single_match_if_loose" works on the list of matches
        ### only when the game ended with a winner.
        ### The difference with the previous procedure is the point of view considered:
        ### the strategy that the software is learning now is defensive (the target is 
        ### not top loose.
        ####################################################################################
        def analyze_single_match_if_loose(match):
            l = len(match)
            if l != 10:     # the length of the match list has to be 10 otherwise error:
                print("Error 6 from class myGameLearning: bad match [",match,"]")
                quit()
            player = match[0]   # the first position of the match list is the first player
            lessons = list()    # Initialize the list of the lesson learnt
            # set some parameters to process the lessons learned:
            if player == CIRCLE:
                idx_start = 1
                myexp = 1
            elif player == STAR:
                idx_start = 2
                myexp = 0
            else:
                print("Error 7 from class myGameLearning: bad player...[",player,"]")
                quit()
            # Evaluate the final status of the board after playing accordingly with the list of the match:
            final_status = [EMPTY for i in range(9)]
            for i in range(1,10):
                if match[i] != None:
                    final_status[match[i]] = (-1)**(i-myexp)
            tris_check = myTris(verbose = False)                # use the class myTris to work on the board and
            (reason,_,_) = tris_check.respond(final_status)     # evaluate the final status of the board after the match
            if reason != "human_victory":
                print("Error 8 from class myGameLearning: bad match final reason...[",match,reason,"]")
                quit()
            # if here the response from myTris class is that the board (at the end of the match) describes a user victory
            for idx in range(idx_start,9,2):
                # analyze every move of the game and get evolution patterns to remember (something like rules or strategies:
                from_status = [EMPTY for i in range(9)]
                for i in range(1,idx):
                    if match[i] != None:
                        from_status[match[i]] = (-1)**(i-myexp)     # define the starting status
                to_status = [EMPTY for i in range(9)]
                for i in range(1,idx+1):
                    if match[i] != None:
                        to_status[match[i]] = (-1)**(i-myexp)       # define the target status
                tris = myTris(verbose = False)                      # use the class myTris to evaluate the board and
                (reason,_,to_status_evaluated) = tris.respond(from_status)  # evaluate the final status of the board after the match
                if to_status_evaluated == to_status or reason == "random_attack":
                    # if myTris class doesn't know how to make a move or if it made a move for loosing
                    # it is possible to create a strategy for defense avoiding the move:
                    weights = [0 for i in range(9)]         # Initialize the list of weights
                    acc = 0                                 # Initialize the counter of the non EMPTY cells that didn't change
                    for j in range(9):
                        if from_status[j] == to_status[j]:
                            if from_status[j] != EMPTY:
                                acc += 1
                    destination_node_id = -1                # Initialize the destination node ID with an impossible value
                                                            # (IDs can be 0 or positive, not negative)
                    for j in range(9):
                        if from_status[j] == CIRCLE and to_status[j] == CIRCLE:     # no status change for CIRCLE: 
                            weights[j] = 1/acc                                      # associate a positive weight 1/acc
                        elif from_status[j] == STAR and to_status[j] == STAR:       # no status change for STAR: 
                            weights[j] = -1/acc                                     # associate a negative weight -1/acc
                        elif from_status[j] == EMPTY and to_status[j] != EMPTY:     # status change from EMPTY to non EMPTY:
                            # find a different destination node ID than the one that is described by the match, that for avoiding loosing:
                            available_cells = list()                        # find the EMPTY board cell list (not considering the cell j)
                            for j1 in range(9):
                                if from_status[j1] == EMPTY and j1 != j:
                                    available_cells.append(j1)
                            if available_cells != []:                       # if there are EMPTY cells:
                                random.shuffle(available_cells)             # randomize the order of the EMPTY cell list
                                destination_node_id = available_cells[0]    # find a new random destination node ID for the move
                        elif from_status[j] == EMPTY and to_status[j] == EMPTY:     # no status change for EMPTY:
                            weights[j] = 0                                          # associate a weight 0
                        else:
                            print("Error 9 from class myGameLearning: bad tris configuration...[",j,from_status,to_status,"]")
                            quit()
                    if destination_node_id >= 0:                        # if the destination node ID is good:
                        lessons.append((weights,destination_node_id))   # add the new lesson learnt to the list  
            return lessons                                              # return the list of all the lessons learnt

        ##############################################################################
        ### The local procedure "save_to_file" saves the lessons learnt (all of them,
        ### existing and new) to a text file.
        ##############################################################################
        def save_to_file(lessons_learnt,my_kb_file_name):
            print("Saving lessons-learnt to output file [",my_kb_file_name,"]...",end="")
            counter = 0
            with open(my_kb_file_name, 'wt') as my_file_handler:
                for (w,j) in lessons_learnt:
                    for val in w:
                        my_file_handler.write("{}\n".format(str(val)))
                    my_file_handler.write("{}\n".format(str(j)))
                    counter += 1
                my_file_handler.close()
            print("done.")
            print("Nr",counter," rules stored.")

        ##############################################################################
        ### The local procedure "load_from_file" load all the already defined lessons  
        ### learnt rules from an existing text file.
        ##############################################################################
        def load_from_file(my_kb_file_name):
            cleaned_list = list()
            if os.path.exists(my_kb_file_name):
                print("Loading lessons-learnt knowledge base from file [",my_kb_file_name,"]...",end="")
                with open(my_kb_file_name, 'rt') as my_file_handler:
                    counter = 0
                    val_list = list()
                    while True:
                        for i in range(9):
                            r = my_file_handler.readline().strip()
                            if not r:
                                break 
                            val_list.append(float(r))
                        r = my_file_handler.readline().strip()
                        if not r:
                            break 
                        cleaned_list.append((val_list,int(r)))
                        val_list = list()
                        counter += 1
                    my_file_handler.close()
                    print("done (",counter,"record loaded)")
                    return cleaned_list
            else:
                return [] 

        if match_status == "win":
            # evaluates the learnt lessons from a match from the winner point of view:
            cleaned_list_for_winning = load_from_file("mytris.lessonslearnt_win.txt")       # get the lessons from the file
            self.lessons_learnt_for_winning = list()
            self.lessons_learnt_for_winning += analyze_single_match_if_win_or_tie(match)    # get the new lessons from the match
            for x in self.lessons_learnt_for_winning:                                       # mix them without repetitions
                if not x in cleaned_list_for_winning:
                    cleaned_list_for_winning.append(x)
            self.lessons_learnt_for_winning = cleaned_list_for_winning
            print("Updating lessons learnt for winning:")
            save_to_file(self.lessons_learnt_for_winning,"mytris.lessonslearnt_win.txt")    # save all the lessons to the same file
                    
        elif match_status == "tie":
            # evaluates the learnt lessons from a match from the tie point of view:
            cleaned_list_for_tie = load_from_file("mytris.lessonslearnt_tie.txt")       # get the lessons from the file
            self.lessons_learnt_for_tie = list()
            self.lessons_learnt_for_tie += analyze_single_match_if_win_or_tie(match)    # get the new lessons from the match
            for x in self.lessons_learnt_for_tie:                                       # mix them without repetitions
                if not x in cleaned_list_for_tie:
                    cleaned_list_for_tie.append(x)
            self.lessons_learnt_for_tie = cleaned_list_for_tie
            print("Updating lessons learnt for tie:")
            save_to_file(self.lessons_learnt_for_tie,"mytris.lessonslearnt_tie.txt")    # save all the lessons to the same file

        elif match_status == "loose":
            # evaluates the learnt lessons from a match from the looser point of view:
            cleaned_list_for_not_loosing = load_from_file("mytris.lessonslearnt_not_loose.txt")     # get the lessons from the file
            self.lessons_learnt_for_not_loosing = list()
            self.lessons_learnt_for_not_loosing += analyze_single_match_if_loose(match)             # get the new lessons from the match
            for x in self.lessons_learnt_for_not_loosing:                                           # mix them without repetitions
                if not x in cleaned_list_for_not_loosing:
                    cleaned_list_for_not_loosing.append(x)
            self.lessons_learnt_for_not_loosing = cleaned_list_for_not_loosing
            print("Updating lessons learnt for not loosing:")
            save_to_file(self.lessons_learnt_for_not_loosing,"mytris.lessonslearnt_not_loose.txt")  # save all the lessons to the same file
            
        else:
            print("Error 10 from class myGameLearning: bad match status...[",match_status,"]")
            quit()

            
            
            
#################
# MAIN PROGRAM: #
#################

if __name__ == '__main__':
    print("Welcome to myTris game:")
    # Create an instance of myTrainedTris class (using basic knowledge + lesson learnt knowledge):
    trained_tris = myTrainedTris()  
    print()
    print("Let's start playing:")
    # Show game board to the user:
    trained_tris.show()
    # Start playing with the user:
    trained_tris.play()
