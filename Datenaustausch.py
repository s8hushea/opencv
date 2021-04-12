from __future__ import absolute_import, division, print_function, unicode_literals

from py_openshowvar import openshowvar

import time
import Utils


client = openshowvar('172.31.1.147', 7000)
#client.can_connect()

# get the actual 7 Polygon pos from camera, actual7PolygonPos is a dictionary
actual7PolygonPos = Utils.getCamPosFromJSON("actual7PolygonPos.json")
print('actual7PolygonPos', actual7PolygonPos)
# get the goal 7 polagon pos from AI Program, goal7PolygonPos is a ditionary
# read actual pos of 7 polygon from actual7PolygonPos.json
goal7PolygonPos = Utils.getGoalPosFromJSON("goal7PolygonPos.json")
#print('actual7PolygonPos', actual7PolygonPos)
# client.write('XPOSITION','{}'.format(round(result[0,0],2)))

client.write('POSITION', '{}'.format(-480), False)
client.write('XNEW', '{}'.format(-360),False)


# X,Y,Z unit mm
frame0 = '{x ' + str(round(actual7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['BigTriangle1'][1])) +',' \
                   +' Z ' + str(371.25) +',' \
                   +' A ' + str(round(0)) +',' \
                   +' B ' + str(round(0)) +',' \
                   +' C ' + str(-180) +'}'

frame1 = '{x ' + str(round(actual7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(actual7PolygonPos['BigTriangle1'][1]) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['BigTriangle1'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['BigTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output True

frame2 = '{x ' + str(round(actual7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['BigTriangle1'][1])) +',' \
                   +' Z ' + str(-9 +100) +',' \
                   +' A ' + str(round(actual7PolygonPos['BigTriangle1'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['BigTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame3 = '{x ' + str(round(goal7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle1'][1])) +',' \
                   +' Z ' + str(-9 +100) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle1'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame4 = '{x ' + str(round(goal7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle1'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle1'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

### Output False

frame5 = '{x ' + str(round(goal7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle1'][1])) +',' \
                   +' Z ' + str(-9+50) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle1'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame6 = '{x ' + str(round(goal7PolygonPos['BigTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle1'][1])) +',' \
                   +' Z ' + str(-9+100) +',' \
                   +' A ' + str(round(actual7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame7 = '{x ' + str(round(actual7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9+100) +',' \
                   +' A ' + str(round(actual7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame8 = '{x ' + str(round(actual7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output True
frame9 = '{x ' + str(round(actual7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9+100) +',' \
                   +' A ' + str(round(actual7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame10 = '{x ' + str(round(goal7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9+100) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame11 = '{x ' + str(round(goal7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output False
frame12 = '{x ' + str(round(goal7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 50) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame13 = '{x ' + str(round(goal7PolygonPos['BigTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['BigTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['BigTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['BigTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame14 = '{x ' + str(round(actual7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle1'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame15 = '{x ' + str(round(actual7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle1'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

#Output True
frame16 = '{x ' + str(round(actual7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle1'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame17 = '{x ' + str(round(goal7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle1'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame18 = '{x ' + str(round(goal7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle1'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output False

frame19 = '{x ' + str(round(goal7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9+ 50) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle1'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle1'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame20 = '{x ' + str(round(goal7PolygonPos['SmallTriangle1'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle1'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame21 = '{x ' + str(round(actual7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame22 = '{x ' + str(round(actual7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

#Output True

frame23 = '{x ' + str(round(actual7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame24 = '{x ' + str(round(goal7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame25 = '{x ' + str(round(goal7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

#Output False
frame26 = '{x ' + str(round(goal7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 50) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame27 = '{x ' + str(round(goal7PolygonPos['SmallTriangle2'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['SmallTriangle2'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['SmallTriangle2'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['SmallTriangle2'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame28 = '{x ' + str(round(actual7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame29 = '{x ' + str(round(actual7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output True
frame30 = '{x ' + str(round(actual7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame31 = '{x ' + str(round(goal7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame32 = '{x ' + str(round(goal7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'
# Output False
frame33 = '{x ' + str(round(goal7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9 + 50) +',' \
                   +' A ' + str(round(goal7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame34 = '{x ' + str(round(goal7PolygonPos['MiddleTriangle'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['MiddleTriangle'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['MiddleTriangle'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['MiddleTriangle'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame35 = '{x ' + str(round(actual7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['Square'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Square'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame36 = '{x ' + str(round(actual7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['Square'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Square'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output True

frame37 = '{x ' + str(round(actual7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['Square'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Square'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame38 = '{x ' + str(round(goal7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['Square'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Square'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame39 = '{x ' + str(round(goal7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['Square'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Square'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output False
frame40 = '{x ' + str(round(goal7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9 + 50) +',' \
                   +' A ' + str(round(goal7PolygonPos['Square'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Square'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame41 = '{x ' + str(round(goal7PolygonPos['Square'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Square'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame42 = '{x ' + str(round(actual7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame43 = '{x ' + str(round(actual7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(actual7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output True

frame44 = '{x ' + str(round(actual7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(actual7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9 + 100) +',' \
                   +' A ' + str(round(actual7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(actual7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame45 = '{x ' + str(round(goal7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9 + 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame46 = '{x ' + str(round(goal7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9) +',' \
                   +' A ' + str(round(goal7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

# Output False
frame47 = '{x ' + str(round(goal7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9 + 50) +',' \
                   +' A ' + str(round(goal7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame48 = '{x ' + str(round(goal7PolygonPos['Parallelgram'][0])) +',' \
                   +' Y ' + str(round(goal7PolygonPos['Parallelgram'][1])) +',' \
                   +' Z ' + str(-9+ 100) +',' \
                   +' A ' + str(round(goal7PolygonPos['Parallelgram'][3])) +',' \
                   +' B ' + str(round(goal7PolygonPos['Parallelgram'][4])) +',' \
                   +' C ' + str(-180) +'}'

frame49 = '{x ' + str(58.83) +',' \
                   +' Y ' + str(660.03) +',' \
                   +' Z ' + str(771.25) +',' \
                   +' A ' + str(-40.93) +',' \
                   +' B ' + str(0) +',' \
                   +' C ' + str(-180) +'}'
#frame1 = '{x -76, Y 882, Z -31, A 90, B 0, C -180}' #BigTriangle1
#frame2 = '{x -76, Y 803, Z 69, A 90, B 0, C -180}' # AZ + 100
#frame3 = '{x 283, Y 803, Z 71, A 0, B 0, C -180}'
#frame4 = '{x 283, Y 803, Z -29, A 0, B 0, C -180}'
# var1 = -300
# var2 = 200
# frame3 = '{x ' + str(var1) +', Y ' + str(var2) +'}'
# print("frame1: ", frame1)
client.write('FRAME0', frame0, False)
client.write('FRAME1', frame1, False)
client.write('FRAME2', frame2, False)
client.write('FRAME3', frame3, False)
client.write('FRAME4', frame4, False)
client.write('FRAME5', frame5, False)
client.write('FRAME6', frame6, False)
client.write('FRAME7', frame7, False)
client.write('FRAME8', frame8, False)
client.write('FRAME9', frame9, False)
client.write('FRAME10', frame10, False)
client.write('FRAME11', frame11, False)
client.write('FRAME12', frame12, False)
client.write('FRAME13', frame13, False)
client.write('FRAME14', frame14, False)
client.write('FRAME15', frame15, False)
client.write('FRAME16', frame16, False)
client.write('FRAME17', frame17, False)
client.write('FRAME18', frame18, False)
client.write('FRAME19', frame19, False)
client.write('FRAME20', frame20, False)
client.write('FRAME21', frame21, False)
client.write('FRAME22', frame22, False)
client.write('FRAME23', frame23, False)
client.write('FRAME24', frame24, False)
client.write('FRAME25', frame25, False)
client.write('FRAME26', frame26, False)
client.write('FRAME27', frame27, False)
client.write('FRAME28', frame28, False)
client.write('FRAME29', frame29, False)
client.write('FRAME30', frame30, False)
client.write('FRAME31', frame31, False)
client.write('FRAME32', frame32, False)
client.write('FRAME33', frame33, False)
client.write('FRAME34', frame34, False)
client.write('FRAME35', frame35, False)
client.write('FRAME36', frame36, False)
client.write('FRAME37', frame37, False)
client.write('FRAME38', frame38, False)
client.write('FRAME39', frame39, False)
client.write('FRAME40', frame40, False)
client.write('FRAME41', frame41, False)
client.write('FRAME42', frame42, False)
client.write('FRAME43', frame43, False)
client.write('FRAME44', frame44, False)
client.write('FRAME45', frame45, False)
client.write('FRAME46', frame46, False)
client.write('FRAME47', frame47, False)
client.write('FRAME48', frame48, False)
client.write('FRAME49', frame49, False)
# client.write('YPOSITION','{}'.format(abs(round(result[1,0],2))))
# client.write('DEGX','{}'.format(0))
# client.write('AVALUE', '{}'.format(alpha))
#  # if(width<height):
#  #     client.write('AVALUE','{}'.format(-40.8))
#  # else:
#  #     client.write('AVALUE','{}'.format(-132.65))
#
# client.write('DEGY','{}'.format(0))
# client.write('DISTANCE','{}'.format(distanceToObject))
message = client.read('STATUS', False).decode('UTF-8')
#message = message.decode('UTF-8')
print(message)

while (client.read('STATUS', False).decode('UTF-8') != 'TRUE'):
    current_pos = client.read('$POS_ACT', False).decode('UTF-8')
    print(current_pos)
    # torque = client.read('$TORQUE_AXIS_ACT', False).decode('UTF-8')
    # print(torque)
    time.sleep(0.5)
output = client.read('POSITION2', False)







