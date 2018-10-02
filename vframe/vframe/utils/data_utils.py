
def ensure_float(n):
  try:
    return float(n)
  except ValueError:
    num, denom = n.split('/')
    try:
      leading, num = num.split(' ')
      whole = float(leading)
    except ValueError:
      whole = 0
    frac = float(num) / float(denom)
    return whole - frac if whole < 0 else whole + frac
