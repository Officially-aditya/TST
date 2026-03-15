# Scientific Calculator - test_calculator.py
# WARNING: Contains intentional structural errors for TST testing

import math

class ScientificCalculator:

    def __init__(self):
        self.history = []
        self.memory = 0

    # STRUCTURAL ERROR 1: Missing self parameter
    def add(a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")  # self not available
        return result

    # STRUCTURAL ERROR 2: Inconsistent indentation
    def subtract(self, a, b):
        result = a - b
            self.history.append(f"{a} - {b} = {result}")  # wrong indent
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    # STRUCTURAL ERROR 3: Missing return on one branch
    def divide(self, a, b):
        if b == 0:
            print("Error: Division by zero")
            # missing return here -- returns None silently
        else:
            result = a / b
            self.history.append(f"{a} / {b} = {result}")
            return result

    # STRUCTURAL ERROR 4: Wrong number of arguments to math function
    def power(self, base, exponent):
        result = math.pow(base, exponent, 2)  # math.pow only takes 2 args
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result

    def square_root(self, n):
        if n < 0:
            raise ValueError("Cannot take square root of negative number")
        result = math.sqrt(n)
        self.history.append(f"sqrt({n}) = {result}")
        return result

    # STRUCTURAL ERROR 5: Calling undefined method
    def logarithm(self, n, base=10):
        if n <= 0:
            raise ValueError("Logarithm undefined for non-positive values")
        result = math.log(n, base)
        self.history.append(f"log({n}) = {result}")
        self.save_to_memory(result)  # method never defined
        return result

    def sine(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        result = math.sin(angle_radians)
        self.history.append(f"sin({angle_degrees}) = {result}")
        return result

    def cosine(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        result = math.cos(angle_radians)
        self.history.append(f"cos({angle_degrees}) = {result}")
        return result

    # STRUCTURAL ERROR 6: Infinite recursion
    def factorial(self, n):
        if n < 0:
            raise ValueError("Factorial undefined for negative numbers")
        if n == 0:
            return 1
        return n * self.factorial(n)  # should be n-1, causes infinite recursion

    def get_history(self):
        if not self.history:
            print("No calculations yet.")
        return self.history

    def clear_history(self):
        self.history = []
        print("History cleared.")


# Entry point
calc = ScientificCalculator()

print(calc.add(5, 3))
print(calc.subtract(10, 4))
print(calc.multiply(6, 7))
print(calc.divide(15, 3))
print(calc.divide(10, 0))
print(calc.power(2, 8))
print(calc.square_root(144))
print(calc.logarithm(100))
print(calc.sine(90))
print(calc.cosine(0))
print(calc.factorial(5))
print(calc.get_history())
