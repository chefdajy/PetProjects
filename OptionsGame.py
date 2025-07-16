import tkinter as tk
import random
import math
import pickle
import os
from scipy.stats import norm



# ========== BLACK SCHOLES OPTIONS PRICING FUNCTION ==========

#Black-Scholes calculation for European option.
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0.0, (S - K) if option_type == 'call' else (K - S))
    d1 = (math.log(S / K) + (r + ((sigma ** 2) / 2)) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)



# ========== DIRECTIONAL LEARNING: QLEARNING ========== (uses pickle)

# Set up of QLearning agent key values for trade decision making.
# Epsilon (probability of choosing random action vs. known rewarding action) is set at 0.3 to start.
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.01
        self.epsilon = epsilon
        self.filename = "q_table.pkl"
        self.load_q_table()

    # Converts the state to a typle (hashable key) for stability.
    def get_state_key(self, state):
        return tuple(state)

    # Retrieves the Q-value for a given state, returning 0 if one is not found.
    def get_q_values(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = {action: 0.0 for action in self.all_actions()}
        return self.q_table[key]

    # Uses the epsilon-greedy strategy to choose an action based on the returned Q-values.
    def choose_action(self, state):
        q_values = self.get_q_values(state)
        if random.random() < self.epsilon:
            return random.choice(list(q_values.keys()))
        return max(q_values, key=q_values.get)

    # Decays epsilon to reduce exploration each round, i.e. less randomness/exploration in action.
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # Updates the Q-table with the reward received.
    def update(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        q_values[action] += self.alpha * (
            reward + self.gamma * max(next_q_values.values()) - q_values[action])
        self.save_q_table()
        self.decay_epsilon()

    # Defines the combinations of all possible actions in option type, time to expiry and strike-spot offsets.
    def all_actions(self):
        return [
            (expiry, offset)
            for expiry in ['7d', '14d', '30d']
            for offset in [-20, -10, 0, 10, 20]
        ]

    # Saves the Q-table (using pickle package).
    def save_q_table(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    # Loads the Q-table (if it exists).
    def load_q_table(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                self.q_table = pickle.load(f)

# Decision making based on the above QLearning system.
def ai_decision_q(agent, price, vol, context_clue):
    sentiment = 1 if 'bullish' in context_clue else -1
    state = [round(price / 10), round(vol * 100), sentiment]
    action = agent.choose_action(state)
    expiry, offset = action
    opt_type = 'call' if sentiment == 1 else 'put'
    strike = round(price + offset)
    return strike, expiry, opt_type, state, action


# ========== PRETRAINING AGENT ==========

# I realised the AI was not performing that well, so pre-training function pre-loading the game up.
# Pretraining the AI Agent with random scenarios so that it is better at the game before starting.
def pretrain_agent(agent, episodes=10000):
    for _ in range(episodes):
        price = random.choice([170, 180, 210])
        vol = random.choice([0.2, 0.5, 0.6])
        simulatedsentiment = random.choice([1, -1])
        state = [round(price / 10), round(vol * 100), simulatedsentiment]


        action = agent.choose_action(state)
        expiry, offset = action
        
        opt_type = 'call' if simulatedsentiment == 1 else 'put'
        strike = price+offset

        final_price = simulate_market(price, simulatedsentiment) 
        next_state = [round(final_price / 10), round(vol * 100), simulatedsentiment]

        T_remain = {"7d": 7 / 365, "14d": 14 / 365, "30d": 30 / 365}
        T = T_remain[expiry]

        # Reward System = Intrinsic Value - Cost (from Black-Scholes)
        intrinsic = max(0, final_price - strike) if opt_type == 'call' else max(0, strike - final_price)
        option_price = black_scholes_price(price, strike, T, 0.01, vol, opt_type)
        reward = (intrinsic - option_price) * 10 if intrinsic > 0 else -option_price
        # Scaling of the rewards helps to make each test more significant to learning.

        agent.update(state, action, reward, next_state)

#To avoid wasting training the agent on market noise, this is a simpler market simulation.
def simulate_market(price, simulatedsentiment):
    price_change = simulatedsentiment * 0.1 * price
    return price + price_change





# ========== GAME CODE ========== (uses tkinter)

# Game Class Setup - Handles UI, game state and trading logic.
class OptionsGame:
    def __init__(self, root, agent):
        self.root = root
        self.root.title("Option Trading Game")
        self.root.configure(bg="black")
        self.r = 0.01
        self.agent = agent
        self.assets = {
            "NVDA": 170,
            "AAPL": 210,
            "GOOGL": 180
        }
        self.vol_options = [0.60, 0.50, 0.20]
        self.expirations = ["7d", "14d", "30d"]
        self.total_pnl = {'player': 0.0, 'ai': 0.0}
        self.pnl_label = None
        self.round = 1
        self.max_rounds = 5
        self.show_start_screen()

    # ---------- START SCREEN ----------

    # Start screen with title, and instructions, and sets round number to 1.
    def show_start_screen(self):
        self.round = 1
        self.clear_screen()

        tk.Label(self.root, text="Welcome to the Options Trading Game!",
             font=("Helvetica", 20, "bold"), bg="black", fg="white").pack(pady=20)

        tk.Button(self.root, text="Start Game", command=self.setup_game,
              font=("Helvetica", 14), bg="white", fg="black").pack(pady=10)

        for header, text in [
        ("CONTEXT:", "You will receive news reports about a company's upcoming earnings, "
                    "including general market sentiment. You'll also see the spot price and volatility. "
                    "Use this information to place either a Call or Put at a fair strike price."),
        ("GOAL:", "Each round, your PnL will be calculated based on actual spot price movements "
                  "and compared with your CPU opponent. The CPU learns using reinforcement learning. "
                  "The winner is the one with the higher total PnL after 5 rounds."),
        ("SUPPLEMENT:", "For assistance, a built-in Black-Scholes calculator is available. "
                       "You may find this useful to determine a fair strike price based on the premium paid for the option.")
        ]:
            tk.Label(self.root, text=header, font=("Helvetica", 14, "bold"),
                 bg="black", fg="white", justify="center").pack(pady=(10, 0))
            tk.Label(self.root, text=text, font=("Helvetica", 14),
                 bg="black", fg="white", wraplength=550, justify="center").pack(pady=(0, 10))

    # ---------- GAME SCREEN + SETUP ----------

    # Game setup for each round, with a random market scenario and list for actions.
    def setup_game(self):
        self.clear_screen()
        tk.Label(self.root, text=f"Round {self.round} of {self.max_rounds}",
                 font=("Helvetica", 14), bg="black", fg="white").pack(pady=5)

        self.asset = random.choice(list(self.assets.keys()))
        self.price = self.assets[self.asset]
        self.vol = random.choice(self.vol_options)
        self.days = random.randint(5, 10)

        self.context_clue = random.choice([
            "Earnings tomorrow! Market sentiment is bullish.",
            "Earnings tomorrow! Market sentiment is bearish."
        ])
        scenario = f"{self.asset} at ${self.price}, Vol = {int(self.vol * 100)}%. {self.context_clue}"

        tk.Label(self.root, text="Market Scenario", font=("Helvetica", 16), bg="black", fg="white").pack(pady=10)
        tk.Label(self.root, text=scenario, font=("Helvetica", 12), bg="black", fg="white").pack(pady=10)

        pnl_frame = tk.Frame(self.root, bg="black")
        pnl_frame.pack(pady=5)

        self.pnl_label = tk.Label(pnl_frame,
                                  text=self.get_pnl_text(),
                                  font=("Helvetica", 12),
                                  fg="cyan", bg="black")
        self.pnl_label.pack(side="left", padx=10)

        tk.Button(pnl_frame, text="Black Scholes Calculator", command=self.open_bs_calculator,
                  bg="white", fg="black").pack(side="left", padx=10)

        self.strikes = [round(self.price + i, 2) for i in range(-50, 51, 10)]

        self.strike_var = tk.DoubleVar(value=self.strikes[0])
        tk.Label(self.root, text="Select Strike", bg="black", fg="white").pack()
        tk.OptionMenu(self.root, self.strike_var, *self.strikes).pack()

        self.expiry_var = tk.StringVar(value=self.expirations[0])
        tk.Label(self.root, text="Select Expiry", bg="black", fg="white").pack()
        tk.OptionMenu(self.root, self.expiry_var, *self.expirations).pack()

        self.option_type = tk.StringVar(value='call')
        tk.Label(self.root, text="Select Option Type", bg="black", fg="white").pack()
        tk.OptionMenu(self.root, self.option_type, "call", "put").pack()

        tk.Button(self.root, text="Submit Choice", command=self.player_submit, bg="white", fg="black").pack(pady=20)

    # Black Scholes Calculator - for if the player chooses to use it.
    def open_bs_calculator(self):
        win = tk.Toplevel(self.root)
        win.title("Black-Scholes Calculator")
        win.configure(bg="black")

        entries = {}

        def make_entry(label_text):
            tk.Label(win, text=label_text, bg="black", fg="white").pack()
            var = tk.StringVar()
            entry = tk.Entry(win, textvariable=var)
            entry.pack()
            entries[label_text] = var

        make_entry("Spot Price (S)")
        make_entry("Strike Price (K)")
        make_entry("Time to Expiry (T in years)")
        make_entry("Volatility (σ)")

        opt_type = tk.StringVar(value='call')
        tk.OptionMenu(win, opt_type, "call", "put").pack()

        result_label = tk.Label(win, text="", bg="black", fg="cyan")
        result_label.pack()

        def calculate():
            try:
                S = float(entries["Spot Price (S)"].get())
                K = float(entries["Strike Price (K)"].get())
                T = eval(entries["Time to Expiry (T in years)"].get())
                # Since the user would enter time in days / 365, this calculates it back to years.
                r = 0.01
                # Fair to leave risk free rate at 0.01 for ease of input to player.
                sigma = float(entries["Volatility (σ)"].get())
                result = black_scholes_price(S, K, T, r, sigma, opt_type.get())
                result_label.config(text=f"Option Price: ${result:.2f}")
            except Exception:
                result_label.config(text="Error in input values.")

        tk.Button(win, text="Calculate", command=calculate, bg="white", fg="black").pack(pady=10)

    def player_submit(self):
        self.player_choice = {
            'strike': self.strike_var.get(),
            'expiry': self.expiry_var.get(),
            'type': self.option_type.get()
        }

        #AI Decision Making via QLearning Agent
        ai_strike, ai_expiry, ai_type, self.ai_state, self.ai_action = ai_decision_q(
            self.agent, self.price, self.vol, self.context_clue)

        self.ai_choice = {
            'strike': ai_strike,
            'expiry': ai_expiry,
            'type': ai_type
        }

        self.run_simulation()

    # Simulation for price movement via geometric Brownian motion, with PnL calculations.
    def run_simulation(self):
        # Initial stock price.
        S_path = [self.price]
        # Time step for daily price movement (252 trading days per year).
        dt = 1 / 252
        for _ in range(self.days):
            # Normally distributed shock.
            shock = random.gauss(0, 1)
            # Drift due to risk-free rate.
            drift = self.r * dt
            # Total price movement = dS=S⋅(drift + volatility x shock x sqrt(dt)).
            dS = S_path[-1] * (drift + self.vol * shock * math.sqrt(dt))
            # Prevent negative prices, with a minimum price of 0.01.
            S_path.append(max(0.01, S_path[-1] + dS))

        final_price = S_path[-1]
        
        # Surprise / Market Noise - 70% chance that surprise matches provided market sentiment.
        if random.random() < 0.70:
            # Suprise element (based on market sentiment), if bullish, (+10-20%), if bearish (-10-25%).
            if "bullish" in self.context_clue:
                surprise = random.uniform(0.10, 0.20)
            else:
                surprise = random.uniform(-0.25, -0.10)
        else:
            surprise = random.uniform(-0.20, 0.20)
        # 30% chance of a random +/- 20% to final price despite market sentiment.

        # Factor in surprise to final price.
        final_price *= (1 + surprise)

        # Converts time to expiry from days into years.
        T_remain = {"7d": 7 / 365, "14d": 14 / 365, "30d": 30 / 365}
        T = T_remain[self.player_choice['expiry']]

        player_payoff = black_scholes_price(final_price, self.player_choice['strike'], T, self.r, self.vol,
                                            self.player_choice['type']) - \
                        black_scholes_price(self.price, self.player_choice['strike'], T, self.r, self.vol,
                                            self.player_choice['type'])

        ai_payoff = black_scholes_price(final_price, self.ai_choice['strike'], T, self.r, self.vol,
                                        self.ai_choice['type']) - \
                    black_scholes_price(self.price, self.ai_choice['strike'], T, self.r, self.vol,
                                        self.ai_choice['type'])

        sentiment = 1 if 'bullish' in self.context_clue else -1
        next_state = [round(final_price / 10), round(self.vol * 100), sentiment]
        self.agent.update(self.ai_state, self.ai_action, ai_payoff, next_state)

        self.total_pnl['player'] += player_payoff
        self.total_pnl['ai'] += ai_payoff
        self.save_total_pnl()

        self.show_results(final_price, player_payoff, ai_payoff)

    # Results screen for each round, with final price movement and PnL calculations.
    def show_results(self, final_price, player_payoff, ai_payoff):
        self.clear_screen()
        tk.Label(self.root, text="Results", font=("Helvetica", 16), bg="black", fg="white").pack(pady=10)

        tk.Label(self.root, text=f"{self.asset} Final Price: ${final_price:.2f}", bg="black", fg="white").pack(pady=5)

        tk.Label(self.root, text=f"You: {self.player_choice['type'].capitalize()} | Strike: {self.player_choice['strike']} | Expiry: {self.player_choice['expiry']}",
                 bg="black", fg="white").pack()

        tk.Label(self.root, text=f"AI: {self.ai_choice['type'].capitalize()} | Strike: {self.ai_choice['strike']} | Expiry: {self.ai_choice['expiry']}",
                 bg="black", fg="white").pack()

        tk.Label(self.root, text=f"Your PnL: ${player_payoff:.2f}",
                 fg="green" if player_payoff > 0 else "red", bg="black").pack(pady=5)

        tk.Label(self.root, text=f"AI PnL: ${ai_payoff:.2f}",
                 fg="green" if ai_payoff > 0 else "red", bg="black").pack(pady=5)

        if self.round >= self.max_rounds:
            tk.Button(self.root, text="Show Final Results", command=self.show_final_results, bg="white", fg="black").pack(pady=10)
        else:
            self.round += 1
            tk.Button(self.root, text="Next Round", command=self.setup_game, bg="white", fg="black").pack(pady=10)

        tk.Button(self.root, text="Play Again", command=self.refresh_and_restart_game, bg="white", fg="black").pack(pady=10)

    # Final results screen for when rounds are complete. Total PnL + Winner/Loser.
    def show_final_results(self):
        self.clear_screen()
        tk.Label(self.root, text="Game Over", font=("Helvetica", 16), bg="black", fg="white").pack(pady=10)

        tk.Label(self.root, text=f"Total Player PnL: ${self.total_pnl['player']:.2f}",
                 fg="green" if self.total_pnl['player'] > 0 else "red", bg="black").pack(pady=5)

        tk.Label(self.root, text=f"Total AI PnL: ${self.total_pnl['ai']:.2f}",
                 fg="green" if self.total_pnl['ai'] > 0 else "red", bg="black").pack(pady=5)

        final_result = "You Win the Game!" if self.total_pnl['player'] > self.total_pnl['ai'] else \
            "AI Wins the Game!" if self.total_pnl['player'] < self.total_pnl['ai'] else "The Game is a Draw."
        tk.Label(self.root, text=final_result, font=("Helvetica", 14, "bold"), bg="black", fg="white").pack(pady=10)

        tk.Button(self.root, text="Play Again", command=self.refresh_and_restart_game, bg="white", fg="black").pack(pady=10)
        tk.Button(self.root, text="Exit Game", command=self.root.destroy, bg="white", fg="black").pack(pady=20)

    # Refresh and restart game at PnL 0.
    def refresh_and_restart_game(self):
        self.total_pnl = {'player': 0.0, 'ai': 0.0}
        self.save_total_pnl()
        self.round = 1
        self.show_start_screen()

    # Clears all remaining widgets from screen.
    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # Returns PnL text for display.
    def get_pnl_text(self):
        return f"Total Player PnL: ${self.total_pnl['player']:.2f} | Total AI PnL: ${self.total_pnl['ai']:.2f}"

    # Saves total PnL using pickle.
    def save_total_pnl(self):
        with open("total_pnl.pkl", "wb") as f:
            pickle.dump(self.total_pnl, f)

    # Loads total PnL (if it exists), just as with the Q Table.
    def load_total_pnl(self):
        if os.path.exists("total_pnl.pkl"):
            with open("total_pnl.pkl", "rb") as f:
                return pickle.load(f)
        return {'player': 0.0, 'ai': 0.0}



# ========== Main Execution ==========

if __name__ == "__main__":
    # Create the QLearning agent instance
    agent_for_game = QLearningAgent()

    # Pretrain this specific agent instance
    print("Starting pretraining...") # Added for clarity
    pretrain_agent(agent_for_game, episodes=10000)
    print("Pretraining complete.") # Added for clarity

    # Initialize the Tkinter root window
    root = tk.Tk()

    # Pass the SAME pre-trained agent instance to the OptionsGame
    app = OptionsGame(root, agent=agent_for_game)
    root.mainloop()
