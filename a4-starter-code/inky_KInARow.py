'''
inky_KInARow.py
Authors: Schmitz, Ilse; Vuu, Alexander; Bui, Elizabeth

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 415, University of Washington
'''

from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Ilse Schmitz, Alexander Vuu, and Elizabeth Bui' 
UWNETIDS = ['inky', 'hocadoo', 'kaitebui']

import time
import random # Used to generate number for twin



UTTERANCE_BANK_WINNING = [
    "You're done Done DONE!",
    "What was that play? You play like Zoey on her forth botte of Soju",
    "Get Ready for my Napalm Era!",
    "Need to beat my face make it cute and savage",
    "Mirror Mirror on the wall, whose the Badest? (Us Hello?)",
    "Knocking you out like a lullaby",
    "Hear that sound ringing in your mind",
    "Even Bobby Makes better plays than you! "
]
UTTERANCE_BANK_LOSING=[
    "Ain't No Saja Boy Gonna Beat Us!",
    "Are You Demons Sent By Gwi-Ma",
    "Gwi-Ma Forces Are Too Strong :c",
    "Dang I was caught off gurad by Jinu's Abs",
    "Ugh! What we're we thinking? I need to carb up after that one",
    "Goo Goo Ga Ga",
    "Wipe that smug look off your face Jinu"
]

class OurAgent(KAgent):  

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Huntrix'
        twin_ver = random.randint(1000, 2000)
        if twin: self.nickname += " " + twin_ver
        self.long_name = 'Huntrix (From K-Pop Demon Hunters)'
        if twin: self.long_name += " " + roman_numeral(twin_ver)
        self.persona = 'Pop-Star'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1

        self.zobrist_current = None # TODO: See if you remove this :/
        self.zobrist_bits = None
        self.transposition_table = [None] * 2**32 # contain all the stored state values :) Always replace

        self.history_cutoff = {}

        self.current_game_type = None
        self.playing_mode = KAgent.DEMO

        self.repeat_count_winning = 0
        self.repeat_count_losing = 0
        self.utt_count_losing = 0
        self.utt_count_winning = 0

    def introduce(self):
        intro = f'\nMy name is {self.long_name}.\n'+\
            '"My creators have charged me with destroying all other agents.\n'+\
            'Don\'t take it personally. ;) \nAlso, I\'m really into K-pop Demon Hunters right now. :D \n'
        if self.twin: intro += "I'm the also the TWIN.\n"
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.

        utterances_matter=True):      # If False, just return 'OK' for each utterance,
                                      # or something simple and quick to compute
                                      # and do not import any LLM or special APIs.
                                      # During the tournament, this will be False..
       if utterances_matter:
           # TODO: Implement K-pop demon hunters utterances. Set global variable / private var to api.
           pass
           # Optionally, import your LLM API here.
           # Then you can use it to help create utterances.
           
       # Write code to save the relevant information in variables
       # local to this instance of the agent.
       ###### Game-type info can be in global variables.
       self.current_game_type = game_type
       self.playing = what_side_to_play
       self.k_to_win = game_type.k
       self.rows = game_type.n
       self.cols = game_type.m

       self.init_zobrist()

       print(f"Prepared to play {game_type.long_name} as {self.playing}.")
       print("Change this to return 'OK' when ready to test the method.")
       return "Not-OK"
   
    def GenNextUtterance(self, best_value):
        
       

        if self.repeat_count > 1: return "Ughhhhhhh Are we still playing??? Can't we wrap things up soon?"
        n = len(UTTERANCE_BANK_WINNING)
        m = len(UTTERANCE_BANK_LOSING)
        if self.utt_count_winning == n:
            self.utt_count = 0
            self.repeat_count_winning += 1
        if self.utt_count_losing == m:
            self.utt_count = 0
            self.repeat_count_losing += 1
        if(best_value >= 0):
            this_utterance = UTTERANCE_BANK_WINNING[self.utt_count]
            self.utt_count_winning += 1
        else:
            this_utterance = UTTERANCE_BANK_LOSING[self.utt_count]
            self.utt_count_losing += 1
        return this_utterance
   

    def make_move(self,
                  current_state,
                  current_remark,
                  time_limit=1000,
                  use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
        """
        Given a valid state, one where there is a possible move and no winner is already established,
        the method finds the best move given the time limit and returns a list of the form
        [[newMove, newState, stat1, ..., stat4], myUtterance].
        
        newMove is of the form (i, j) giving the row and column positions where the player is putting an X or O.
        newState is a valid state object representing the result of making the move.
        myUtterance is the player's current contribution to the ongoing dialog and must be a string.
        """
        print("make_move has been called")

        # print("code to compute a good move should go here.")

        # Call minimax to find the best move.
        best_move, best_value = self.minimax(current_state,
                                             max_ply,
                                             time_limit,
                                             use_alpha_beta)
        
        # Create new state using the best_move
        new_state = State(old=current_state)
        i = best_move[0]
        j = best_move[1]
        new_state.board[i][j] = current_state.whose_move
        new_state.whose_move = "O" if current_state.whose_move is "X" else "X"

        self.zobrist_current = self.hash(self.zobrist_current, new_state.whose_move, (i,j))

        nextUtterance = self.GenNextUtterance(best_value)

        
        # return [[best_Move, newState, stat1, ..., stat4], myUtterance]
        return [[best_move, new_state], nextUtterance] # TODO: Need to return the statistics??? For zobrist hashing :/ 

   
    def minimax(self,
                state,
                depth_remaining,
                time_limit,
                pruning=False,
                alpha=None,
                beta=None,
                zhash=None):
        """
        minimax is a helper function called by make_move. It implements minimax search.
        Utilizes history heuristic and transposition table to increase efficency. 
        Returns a move an float value that corresponds to the evaluation of the move.
        """

        ## TODO: Use the history heuristic next :)
        ### TODO: Add the values state evaluation into the TT. 
        # print("Calling minimax. We need to implement its body.")
        
        start_time = time.time()

        # If at max_ply, return static evaluation of state.
        if depth_remaining == 0:
            return (None, self.static_eval(state, self.current_game_type))
        
        best_value = float("inf") if (player == "O") else float("-inf")
        
        zhash = self.zobrist_current if zhash is None else zhash
        TT_hash = zhash & 0xFFFFF
        if self.transposition_table[TT_hash][0] == self.zobrist_hash \
            and self.transposition_table[TT_hash][4] == state.whose_move:
            
            # If exact, return the best move.
            if self.transposition_table[TT_hash][2] == 0:
                return (self.transposition_table[TT_hash][3], self.transposition_table[TT_hash][1]) # (best_move, value)
            
            # If non-exact, use it to tighten the bounds.
            if self.transposition_table[TT_hash][2] == 2 and state.whoose_move == "O":
                best_value = self.transposition_table[TT_hash][1]

            if self.transposition_table[TT_hash][2] == 1 and state.whoose_move == "X":
                best_value = self.transposition_table[TT_hash][1]


        # Otherwise, find the successors and corresponding moves.
        successors, moves = successors_and_moves(state)
        successors, moves = self.sort_by_history(successors, moves, zhash)

        # If no legal moves, return the static evaluation of state.
        if not successors:
            return (None, self.static_eval(state, self.current_game_type))

        ### EDIT WORK FROM HERE

        # Initialize best value depending on whose turn it is
        player = state.whose_move
        best_move = None

        cutoff_type = None

        for i in range(len(successors)):
            child = successors[i]
            move = moves[i]
            child_zhash = self.hash(zhash, child.whoose_move, move)
            # Check if the time limit has been reached
            check_time = time.time()
            elapsed = check_time - start_time
            if elapsed >= time_limit:
                break
            time_limit = time_limit - elapsed
            # recusion on minimax
            _, value = self.minimax(child,
                                    depth_remaining - 1,
                                    time_limit,
                                    alpha, beta, child_zhash)

            if (player == "O") & (value < best_value):
                best_value = value
                best_move = moves[i]
            elif (player == "X") & (value > best_value):
                best_value = value
                best_move = moves[i]

            # Alpha-beta pruning
            if pruning:
                if alpha is not None and beta is not None:
                    if player == "O":
                        beta = min(beta, best_value)
                    else:
                        alpha = max(alpha, best_value)
                    if beta <= alpha:
                        if child_zhash in self.history_cutoff:
                            self.history_cutoff[child_zhash] += 1
                        else:
                            self.history_cutoff[child_zhash] = 1

                        cutoff_type = 2 if player == "O" else 1
                        break
        
        self.update_TT(self, TT_hash, best_value, cutoff_type, best_move, player)
        return (best_move, best_value)

        # default_score = 0 # Value of the passed-in state. Needs to be computed.
    
        # return [default_score, next_move, "my own optional stuff"]
        # Only the score is required here but other stuff can be returned
        # in the list, after the score, in case you want to pass info
        # back from recursive calls that might be used in your utterances,
        # etc. 

    def sort_by_history(self, successors, moves, parent_hash):
        priority = []
        for i in range in len(successors):
            child_zhash = self.hash(parent_hash, successors[i].whoose_move, moves[i])
            if child_zhash in self.history_cutoff:
                priority.append(self.history_cutoff[child_zhash])
            else:
                child_zhash.append(0)
        
        return zip(*zip(successors, moves, priority).sort(key=lambda tri: tri[2]))

 

    def score_lines(self, single_line, symbols_for_win):
            if('-' in single_line):
                return 0
            for i in range(symbols_for_win):
                if(single_line.count("X") == (symbols_for_win-i) and single_line.count(" ") == i):
                    return 10**(symbols_for_win-i-1)
                if(single_line.count("O") == (symbols_for_win-i) and single_line.count(" ") == i):
                    return -1*10**(symbols_for_win-i-1)
            if(single_line.count(" ") == len(single_line)):
                        return 0
                
            # raise ValueError("could not identify score for a line")
            return 0
    
    def static_eval(self, state, game_type=None):
        # print('calling static_eval. Its value needs to be computed!')
        # Values should be higher when the states are better for X,
        # lower when better for O.

        check_state = State(old=state)

        
        if game_type is None:
            game_type = self.current_game_type

        
        rows = self.rows
        cols = self.cols
        k = self.k_to_win
        
        if(k > cols and k > rows):
            raise ValueError("symbols for a win is greater rows or columns")
        
        #all possible lines to score
        lines = []

        #horizontal lines
        horizontalLines = []
        horizontalLines.extend(check_state.board)
        if(rows > k):
            for splitline in horizontalLines:
                for i in range(rows-k+1):
                    lines.append(splitline[i : i+k])

        #vertical lines
        
        for c in range(cols):
            if(cols == k):
            #lines.append([check_state.board[r][c] for r in range(rows)])
                lines.append([check_state.board[r][c] for r in range(rows)])

            else: #if(cols > k):
                verticalLines = []
                verticalLines.append([check_state.board[r][c] for r in range(rows)])
                for splitline in verticalLines:
                    for i in range(cols-k+1):
                        lines.append(splitline[i : i+k])
            
        #diagonal lines left to right
        for r in range(rows-k+1):# r Row
            for c in range(cols-k+1):# c Column
                linemaker = []
                for l in range(k): # l Length
                    linemaker.append(check_state.board[r+l][c+l])

                lines.append(linemaker)
        #diagonal lines  right to left
        for r in range(rows-1, k-2, -1):# r Row
            for c in range(cols-1, k-2, -1):# c Column
                linemaker = []
                for l in range(k): # l Length
                    linemaker.append(check_state.board[r-l][c-l])

                lines.append(linemaker)
            

        total = 0
        for LineToScore in lines:
            
            total += self.score_lines(LineToScore, k)
            

        # print("its " + str(self.nickname) + "'s score after making a move.")
        return total
    
    def init_zobrist(self):
        """
        Generates a table of random bitstrings for every possible move on the board.
        There are 2 pieces ('X' and 'O') as well as n x m possible positions.
        Hence, we generate 2 x n x m possible bit strings.
        """

        self.zobrist_bits = [[None] * self.rows*self.cols] * 2

        for piece in range(2): # Loop over pieces
            for position in range(self.rows*self.cols): # Loop over all board positions
                self.zobrist_bits[piece][position] = random.getrandbits(64)
        self.zobrist_current = 0
    
    def hash(self, state_hash, piece, position: tuple[int, int]):
        """
        Generate iterative hash code given the hash code of the existing state, the piece to be
        added, and the position where it will be added.
        """
        table_position = position[0] * self.rows + position[1]
        piece = 0 if piece == "O" else 1
        return state_hash ^ self.zobrist_bits[piece][table_position]
    
    def hash(self, state):
        """
        Compute the hash for a state.
        """
        h = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if state[i][j] == "X" or state.board[i][j] == "O":
                    piece = 0 if state.board[i][j] == "O" else 1
                    table_position = i * self.rows + j
                    h = h ^ self.zobrist_bits[piece][table_position]
        return h
    
    def update_TT(self, hash_state, score, cutoff_type, best_move, player):
        """
        Update the transition table to include the overide.
        Cut-off type should be an integer, 0, 1, 2 where 0 represents "exact value", 1 - "alpha cut-off", and 2 - "beta cut-off".
        """
        TT_hash = hash_state & 0xFFFFF
        self.transposition_table[TT_hash] = (hash_state, score, cutoff_type, best_move, player)
 
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

# For example, other("X") = "O".
def other(p):
    if p=='X': return 'O'
    return 'X'

# Randomly choose a move.
def chooseMove(statesAndMoves):
    states, moves = statesAndMoves
    if states==[]: return None

    # random_index = randint(0, len(states)-1)
    random_index = 0
    my_choice = [states[random_index], moves[random_index]]
    return my_choice

def move_gen(state):
    b = state.board
    p = state.whose_move
    o = other(p)
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            news = do_move(state, i, j, o)
            yield [(i, j), news]

# This uses the generator to get all the successors.
def successors_and_moves(state):
    moves = []
    new_states = []
    for item in move_gen(state):
        moves.append(item[0])
        new_states.append(item[1])

    return [new_states, moves]

# Perform a a move to get a new state.
def do_move(state, i, j, o):
            news = State(old=state)
            news.board[i][j] = state.whose_move
            news.whose_move = o
            return news

def roman_numeral(n):
    """ 
    Method takes an integer between 1000 to 2000 and returns the roman numeral representation.
    """
    romans = [(1, "I"), (5, "V"), (10, "X"), (50, "L"), (100, "C"), (500, "D"), (1000, "M")]
    
    i = 7
    final = ""

    while n > 0:
        if n - romans[7][0] > 0:
            final += romans[7][1] + final
            n = n - romans[7][0]
        else:
            i -= 1
    return final
