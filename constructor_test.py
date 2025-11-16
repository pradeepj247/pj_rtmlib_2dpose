
from rtmlib import Body
import inspect

print("Body constructor parameters:")
print(inspect.signature(Body.__init__))

# Let's also check the docstring if available
print("\\nDocstring:")
print(Body.__init__.__doc__)
