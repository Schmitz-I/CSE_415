from hocadoo_KInARow import OurAgent
from game_types import TTT, FIAR, Cassini, State

def print_board(state):
    for row in state.board:
        print(row)
    print()

# --- Test 1: Tic-Tac-Toe ---
agent = OurAgent()

# # Prepare the agent with the game type
# agent.prepare(TTT, "X", "Opponent", utterances_matter=False)

# # Copy the initial state
# state = State(old=TTT.initial_state)

# # Make a few moves for testing
# state.board[0][0] = "X"
# state.board[0][1] = "X"
# state.board[1][1] = "O"

# print("TTT Test State:")
# print_board(state)

# # Run static evaluation
# score = agent.static_eval(state, TTT)
# print("Static Evaluation Score:", score)

# # Test score_lines directly
# line = ["X", "X", " "]
# print("Score for line ['X','X',' ']:", agent.score_lines(line, 3))


# --- Test 2: Five-in-a-Row (FIAR) ---
agent.prepare(FIAR, "X", "Opponent", utterances_matter=False)
fiar_state = State(old=FIAR.initial_state)

# Make a few sample moves
fiar_state.board[1][1] = "X"
fiar_state.board[1][2] = "X"
fiar_state.board[1][3] = "X"
fiar_state.board[1][4] = " " 

print("\nFIAR Test State:")
print_board(fiar_state)

score_fiar = agent.static_eval(fiar_state, FIAR)
print("Static Evaluation Score (FIAR):", score_fiar)




# # --- Test 3: Cassini ---
# agent.prepare(Cassini, "X", "Opponent", utterances_matter=False)
# cassini_state = State(old=Cassini.initial_state)

# # Sample move
# cassini_state.board[3][3] = "X"
# cassini_state.board[3][4] = "X"
# cassini_state.board[3][5] = "X"

# print("\nCassini Test State:")
# print_board(cassini_state)

# score_cassini = agent.static_eval(cassini_state, Cassini)
# print("Static Evaluation Score (Cassini):", score_cassini)
