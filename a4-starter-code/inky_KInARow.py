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
import random # Used to generate number for twin, bit-wise strings for hashing.


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
        # Agent Info
        self.twin=twin
        self.nickname = 'Huntrix'#'Slim Schady'
        twin_ver = random.randint(1000, 2000)
        if twin: self.nickname += " " + twin_ver
        self.long_name = 'Huntrix (From K-Pop Demon Hunters)' #'Schadenfreude Collector'
        if twin: self.long_name += " " + roman_numeral(twin_ver)
        self.persona = 'Pop-Star'#'Schady'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}

        # Game info
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.current_game_type = None
        self.playing_mode = KAgent.DEMO

        # Game stats
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.zobrist_table_num_writes_this_turn = -1
        self.zobrist_table_num_read_this_turn = -1

        self.repeat_count_winning = 0
        self.repeat_count_losing = 0
        self.utt_count_losing = 0
        self.utt_count_winning = 0

        # Zobrist hashing variables
        self.zobrist_bits = None
        self.transposition_table = [None] * 2**20 # contain all the stored state values :) Never replace
        self.zobrist_current = None

        # Ordering heuristic for alpha-beta pruning
        self.history_cutoff = {}


    def introduce(self):
        """
        Returns string introduction for agent. 
        """
        intro = f'\nMy name is {self.long_name}.\n'+\
            '"My creators have charged me with destroying all other agents.\n'+\
            'Don\'t take it personally. ;) \nAlso, I\'m really into K-pop Demon Hunters right now. :D" \n'
        if self.twin: intro += "I'm the also the TWIN.\n"
        return intro

    
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1,
        utterances_matter=True): 
        """
        Receives and acknowledges game information from the game master. 
        """
        if utterances_matter:
           pass
        
        self.current_game_type = game_type
        self.playing = what_side_to_play
        self.k_to_win = game_type.k
        self.rows = game_type.n
        self.cols = game_type.m

        self.init_zobrist()

        print(f"Prepared to play {game_type.long_name} as {self.playing}.")
        return "OK"
   
    def gen_next_utterance(self, best_value):
        """
        This function chooses the next utterance.
        """

        if self.repeat_count_winning > 1: return "Ughhhhhhh Are we still playing??? Can't we wrap things up soon?"

        n = len(UTTERANCE_BANK_WINNING)
        m = len(UTTERANCE_BANK_LOSING)

        if self.utt_count_winning == n:
            self.utt_count_winning = 0
            self.repeat_count_winning += 1

        if self.utt_count_losing == m:
            self.utt_count_losing = 0
            self.repeat_count_losing += 1

        if(best_value >= 0):
            this_utterance = UTTERANCE_BANK_WINNING[self.utt_count_winning]
            self.utt_count_winning += 1

        else:
            this_utterance = UTTERANCE_BANK_LOSING[self.utt_count_losing]
            self.utt_count_losing += 1

        return this_utterance
   

    def make_move(self,
                  current_state,
                  current_remark,
                  time_limit=1000,
                  use_alpha_beta=True,
                  use_zobrist_hashing=True, max_ply=3,
                  special_static_eval_fn=None):
        """
        Given a valid state, one where there is a possible move and no winner is already established,
        the method finds the best move given the time limit and returns a list of the form
        [[newMove, newState, stat1, ..., stat4], myUtterance].
        
        newMove is of the form (i, j) giving the row and column positions where the player is putting an X or O.
        newState is a valid state object representing the result of making the move.
        myUtterance is the player's current contribution to the ongoing dialog and must be a string.
        """
        print(f"{self.nickname} is thinking.")

        # Calculate the zobrist hashing for the state
        if use_zobrist_hashing:
            self.zobrist_current = self.complete_hash(current_state)

        # Reset stat values
        self.alpha_beta_cutoffs_this_turn = 0
        self.num_static_evals_this_turn = 0
        self.zobrist_table_num_writes_this_turn = 0
        self.zobrist_table_num_read_this_turn = 0
        self.zobrist_table_num_hits_this_turn = 0

        # Call minimax to find the best move.
        best_move, best_value = self.minimax(current_state,
                                             max_ply,
                                             time_limit,
                                             use_alpha_beta,
                                             use_zobrist_hashing)
        
        # Create new state using the best_move
        new_state = State(old=current_state)
        i = best_move[0]
        j = best_move[1]
        new_state.board[i][j] = current_state.whose_move
        new_state.whose_move = "O" if current_state.whose_move == "X" else "X"

        nextUtterance = self.gen_next_utterance(best_value)

        print(f"{self.nickname} is making it's move.")

        print("Stats:", "cutoffs", self.alpha_beta_cutoffs_this_turn, "static evals",
               self.num_static_evals_this_turn, "table writes", self.zobrist_table_num_writes_this_turn, 
               "table reads", self.zobrist_table_num_read_this_turn, "table hits", self.zobrist_table_num_hits_this_turn)

        if use_alpha_beta and use_zobrist_hashing:
            return [[best_move, new_state#,
                #self.alpha_beta_cutoffs_this_turn,
                # self.num_static_evals_this_turn, 
                #self.zobrist_table_num_writes_this_turn,
                #self.zobrist_table_num_read_this_turn,
                #self.zobrist_table_num_hits_this_turn
                ], nextUtterance]
        return [[best_move, new_state], nextUtterance]

   
    def minimax(self,
                state,
                depth_remaining,
                time_limit,
                pruning=True,
                hashing=True,
                alpha=None,
                beta=None,
                zhash=None):
        """
        minimax is a helper function called by make_move. It implements minimax search.
        Utilizes history heuristic and transposition table to increase efficency. 
        Returns a move an float value that corresponds to the evaluation of the move.
        """
        
        start_time = time.time()

        # If at max_ply, return static evaluation of state.
        if depth_remaining == 0:
            return (None, self.static_eval(state, self.current_game_type))
        
        # Initialize values
        player = state.whose_move
        best_move = None
        best_value = float("inf") if (player == "O") else float("-inf")
        cutoff_type = 0 # To be stored in zobrist hash table
        
        # See if the state has been hashed in the TT
        zhash = self.zobrist_current if zhash is None else zhash
        TT_hash = zhash & 0xFFFFF

        # Update number of reads
        self.zobrist_table_num_read_this_turn += 1

        # If state is stored in transposition table
        if hashing and self.transposition_table[TT_hash] is not None and self.transposition_table[TT_hash][0] == zhash:
            
            self.zobrist_table_num_hits_this_turn += 1

            flag = self.transposition_table[TT_hash][2]
            stored_value = self.transposition_table[TT_hash][1]
            stored_best_move = self.transposition_table[TT_hash][3]

            # If exact, return the best move.
            if flag == 0:
                return (stored_best_move, stored_value) 
            
            # If non-exact, use it to tighten the bounds.


            # Tighten alpha bound: Stored value is a lower bound (>= value)
            if flag == 1:
                alpha = max(alpha, stored_value) if alpha is not None else stored_value

                if pruning and beta is not None and alpha >= beta:
                    # Cutoff found from TT. Stop search immediately.
                    self.alpha_beta_cutoffs_this_turn += 1
                    return (stored_best_move, stored_value)

            # Tighten beta bound: Stored value is a upper bound (<= value)
            if flag == 2:
                beta = min(beta, stored_value) if beta is not None else stored_value

                if pruning and alpha is not None and alpha >= beta:
                    # Cutoff found from TT. Stop search immediately.
                    self.alpha_beta_cutoffs_this_turn += 1
                    return (stored_best_move, stored_value)


        # Otherwise, find the successors and corresponding moves.
        successors, moves = successors_and_moves(state)

        # If no legal moves, return the static evaluation of state.
        if not successors:
            return (None, self.static_eval(state, self.current_game_type))

        # Sort by history heuristic
        successors, moves = self.sort_by_history(successors, moves, zhash)

        # Evaluate successors
        for i in range(len(successors)):
            child = successors[i]
            move = moves[i]
            child_zhash = self.hash(zhash, player, move)

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
                                    pruning,
                                    hashing,
                                    alpha, beta, child_zhash)

            if (player == "O") & (value < best_value):
                best_value = value
                best_move = moves[i]
            
            elif (player == "X") & (value > best_value):
                best_value = value
                best_move = moves[i]

            # Update alpha and beta
            if player == "O":
                beta = min(beta, best_value) if beta is not None else best_value
            else:
                alpha = max(alpha, best_value) if alpha is not None else best_value
            
            # Alpha-beta pruning
            if pruning:
                if alpha is not None and beta is not None:
                    if beta <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        if child_zhash in self.history_cutoff:
                            self.history_cutoff[child_zhash] += 1
                        else:
                            self.history_cutoff[child_zhash] = 1

                        cutoff_type = 2 if player == "O" else 1
                        self.update_TT(child_zhash, best_value, cutoff_type, best_move)
                        break
        
        self.update_TT(TT_hash, best_value, cutoff_type, best_move)
        return (best_move, best_value)

    def sort_by_history(self, successors, moves, parent_hash):
        """
        User the history heuristic to order successor nodes for alpha-beta pruning.
        The history heuristic orders based off the number of cut-offs a state has
        created at any depth.
        """
        priority = []

        for i in range(len(successors)):
            child_zhash = self.hash(parent_hash, successors[i].whose_move, moves[i])
            if child_zhash in self.history_cutoff:
                priority.append(self.history_cutoff[child_zhash])
            else:
                priority.append(0)
        scored_triples = list(zip(successors, moves, priority))
        scored_triples.sort(key=lambda tri: tri[2], reverse=True)
        sorted_successors, sorted_moves, _ = zip(*scored_triples)
        
        return sorted_successors, sorted_moves
 

    def score_lines(self, single_line, symbols_for_win):
        """
        Helper function for the static evaluation function. Returns the value of a given line.
        """
        if('-' in single_line):
            return 0
        
        for i in range(symbols_for_win):
            if(single_line.count("X") == (symbols_for_win-i) and single_line.count(" ") == i):
                return 10**(symbols_for_win-i-1)
            
            if(single_line.count("O") == (symbols_for_win-i) and single_line.count(" ") == i):
                return -1*10**(symbols_for_win-i-1)
            
        if(single_line.count(" ") == len(single_line)):
            return 0
                
        return 0
    

    def static_eval(self, state, game_type=None):
        """
        Static Evaluation function.
        """
        check_state = State(old=state)

        if game_type is None:
            game_type = self.current_game_type

        rows = game_type.n
        cols = game_type.m
        k = game_type.k # symbols to win
        
        if(k > cols and k > rows):
            raise ValueError("symbols for a win is greater rows or columns")
        
        # all possible lines to score
        lines = []

        # horizontal lines
        lines.extend(check_state.board)

        # vertical lines
        for c in range(cols):
            lines.append([check_state.board[r][c] for r in range(rows)])
        
        # diagonal lines left to right
        for r in range(rows-k+1):# r Row
            for c in range(cols-k+1):# c Column
                linemaker = []
                for l in range(k): # l Length
                    linemaker.append(check_state.board[r+l][c+l])
                lines.append(linemaker)
        
        # diagonal lines right to left
        for r in range(rows-1, k-2, -1): # Iterate over rows
            for c in range(cols-1, k-2, -1): # Iterate over columns

                linemaker = []

                for l in range(k): # Iterate over length
                    linemaker.append(check_state.board[r-l][c-l])
                lines.append(linemaker)
            
        total = 0

        for LineToScore in lines:
            total += self.score_lines(LineToScore, k)
        
        # Add 1 to the number of static evaluations this turn.
        self.num_static_evals_this_turn += 1

        return total
    

    ### Zobrist Hashing Functions

    def init_zobrist(self):
        """
        Generates a table of random bitstrings for every possible move on the board.
        There are 2 pieces ('X' and 'O') as well as n x m possible positions.
        Hence, we generate 2 x n x m possible bit strings.
        """

        self.zobrist_bits = [[None] * (self.rows * self.cols) for _ in range(2)]

        for piece in range(2): # Loop over pieces
            for position in range(self.rows*self.cols): # Loop over all board positions
                self.zobrist_bits[piece][position] = random.getrandbits(64)
        self.zobrist_current = 0
    

    def hash(self, state_hash, piece, position: tuple[int, int]):
        """
        Generate iterative hash code given the hash code of the existing state, the piece to be
        added, and the position where it will be added.
        """
        table_position = position[0] * self.cols + position[1]
        
        piece = 0 if piece == "O" else 1

        return state_hash ^ self.zobrist_bits[piece][table_position] #^ self.zobrist_turn
    
    def complete_hash(self, state):
        """
        Generate the hash of a state.
        """
        hash = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if state.board[i][j] != " " and state.board[i][j] != "-":
                    table_position = i * self.cols + j
                    piece = 0 if state.board[i][j] == "O" else 1
                    hash ^= self.zobrist_bits[piece][table_position]

        return hash #^ self.zobrist_turn

    
    def update_TT(self, hash_state, score, cutoff_type, best_move):
        """
        Update the transition table to include the overide.
        Cut-off type should be an integer, 0, 1, 2 where 0 represents "exact value", 1 - "alpha cut-off", and 2 - "beta cut-off".
        """
        TT_hash = hash_state & 0xFFFFF
        
        self.transposition_table[TT_hash] = (hash_state, score, cutoff_type, best_move)
        self.zobrist_table_num_writes_this_turn += 1


### HELPER FUNCTIONS ###

def move_gen(state):
    """
    This function generates legal moves, and their corresponding states.
    """
    b = state.board
    p = state.whose_move
    o = "O" if p != "O" else "X"
    mCols = len(b[0])
    nRows = len(b)

    for i in range(nRows):
        for j in range(mCols):
            if b[i][j] != ' ': continue
            news = do_move(state, i, j, o)
            yield [(i, j), news]


def successors_and_moves(state):
    """
    Utilizes move_gen to compile a list of all legal successor states and the corresponding move.
    """
    moves = []
    new_states = []
    for item in move_gen(state):
        moves.append(item[0])
        new_states.append(item[1])

    return [new_states, moves]


def do_move(state, i, j, o):
    """
    Helper method for move_gen. Returns the corresponding state after a move.
    """
    news = State(old=state)
    news.board[i][j] = state.whose_move
    news.whose_move = o
    return news


def roman_numeral(n):
    """ 
    Method takes an integer between 1000 to 2000 and returns the roman numeral representation.
    """
    romans = [
        (2000, "MM"), (1900, "MCM"), (1500, "MD"), (1400, "MCD"),
        (1000, "M"),
        (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
        (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
        (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]
    
    final = ""

    for val, sym in romans:
        while n >= val:
            final += sym
            n = n - val
    return final

