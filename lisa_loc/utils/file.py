import re

def numerical_sort_key(file_name):
  """Extracts the numeric prefix from a filename and returns it as an integer."""
  match = re.match(r'(\d+)\.pcd', file_name)
  if match:
    return int(match.group(1))
  else:
    return 0 