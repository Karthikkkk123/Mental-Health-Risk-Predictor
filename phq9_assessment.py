def get_response(question):
    options = {
        "0": "Not at all",
        "1": "Several days",
        "2": "More than half the days",
        "3": "Nearly every day"
    }
    print("\n" + question)
    for key, value in options.items():
        print(f"{key}: {value}")
    
    while True:
        choice = input("Enter your choice (0-3): ")
        if choice in options:
            return int(choice)
        else:
            print("Invalid input. Please enter a number between 0 and 3.")

def interpret_score(score):
    if score <= 4:
        return "Minimal or none"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately severe"
    else:
        return "Severe"

def main():
    print("PHQ-9 Depression Assessment Tool\n")
    
    questions = [
        "1. Little interest or pleasure in doing things",
        "2. Feeling down, depressed, or hopeless",
        "3. Trouble falling or staying asleep, or sleeping too much",
        "4. Feeling tired or having little energy",
        "5. Poor appetite or overeating",
        "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
        "7. Trouble concentrating on things, such as reading the newspaper or watching television",
        "8. Moving or speaking so slowly that other people could have noticed — or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
        "9. Thoughts that you would be better off dead or of hurting yourself in some way"
    ]
    
    total_score = sum(get_response(q) for q in questions)
    severity = interpret_score(total_score)
    
    print(f"\nYour PHQ-9 Total Score: {total_score}")
    print(f"Depression Severity: {severity}")

if __name__ == "__main__":
    main()
