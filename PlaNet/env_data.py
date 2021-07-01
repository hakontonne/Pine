



official_envs = {
    'cartpole-balance'       : ['Balance an unactuated pole by applying forces to a cart at its base, starting with the pole upwards and with non-sparse rewards'],
    'finger-spin'            : ['A 3DoF planar finger is required to rotate a toy body on an unactuated hinge, the body must be continually rotated or spun'],
    'walker-walk'            : ['A planar walker with two legs and a torso, recieve rewards for standing the torso upright and in a certain high, while also walking'],
    #'walker-stand'           : ['A planar walker with two legs and a torso, recieve rewards for standing the torso upright and in a certain high'],
    'walker-run'             : ['A planar walker with two legs and a torso, recieve rewards for standing the torso upright and in a certain high, while also running fast'],
    'reacher-easy'           : ['A simple two link planar reacher, get rewards when the end effector is inside the large target area'],
    'cheetah-run'            : ['A planar runner biped, based on a simplified cheetah, receive rewards for running quickly forwards'],
    'acrobot-swingup'        : ['The underactuated double pendulum, torque applied to the second joint, where the goal is to swing up and balance the pendulum'],
    'ball_in_cup-catch'      : ['An actuated planar receptacle can translate in the vertical plane in order to swing and catch a ball attached to its bottom, recieving sparse rewards when the ball is in the receptacle'],
    'pendulum-swingup'       : ['Classic pendulum swingup task, with torque a sixth of whats required to lift the pendulum, swing the pendulum up and balance it, receive sparse rewards while the pendulum is within 30 degrees of the vertical'],
    'hopper-stand'           : ['A planar one-legged hopper, initialised with a random configuration, make the hopper stand up and bring the torso into a minimal height'],
    'hopper-hop'             : ['A planar one-legged hopper, initialised with a random configuration, make the hopper hop forward while also stadning up'],
    'manipulator-insert_ball': ['A planar manipulator is rewarded for bringing a ball into a basket'],
    'manipulator-bring_peg'  : ['A planar manipulator is rewarded for bringing a peg into another peg'],
    #'manipulator-insert_peg' : ['A planar manipulator is rewarded for bringing a peg into a slot']

}


other_envs = {
    'cartpole-balance'      : ['A pole attached to cart, move the cart to balance the pole above the cart, starting with the pole upright',
                                'Move a cart with unpowered pole attached and balance the pole upright, with the pole starting in upright',
                                'Balance a pole attached to a cart by moving the cart right or left, starting with the pole in an upright position'],

    'finger-spin'           : ['Using a planer finger, rotate a spinner and dont stop spinning it',
                               'Rotate a toy using a planar finger manipulator and continue to spin the toy',
                               'An hinged and unpowered toy should be spun using a finger planar'],

    'walker-walk'           :['Make a planar walker with two legs stand up and walk',
                              'Using a two-legged walker, stand up and walk the walker, not going to fast',
                              'Walk a planar walker with two legs'],

    'walker-run'            :['A planar walker should stood upright and then make run as fast as possible',
                              'Stand a two-legged walker upright and make it run',
                              'Make the walker stand up and then run as fast as possible'],

    'reacher-easy'          :['Using a two link reacher robot, reach the red target area with the tip of the maniupulator',
                              'Place the tip of the reacher robot on the red target area',
                              'The reacher manipulator has an end effector that should be placed on the red target area'],

    'cheetah-run'           :['Make the cheetah based biped run as fast as possible',
                              'Run as fast as possible with the planar cheetah runner',
                              'A two legged cheetah-based planar robot, make it run as fast as possible'],

    'acrobot-swingup'       :['A two linked version of the classic pendulum problem, apply forces on the second joint to make the pendulum swing up and balance',
                              'Make the two linked pole swing up and then balance it',
                              'Swing the two linked pendulum up and balance it there'],

    'ball_in_cup-catch'     :['Catch the ball hanging underneath the cup and keep it inside the cup to get rewards',
                              'Using a planar cup, catch the ball hanging underneath it by swinging the ball up and catching it, get rewards when its inside the cup',
                              'An unactuated ball is hanging beneath a planar cup, swing the ball up and catch it with the cup and keep it there to receive rewards'],

    'pendulum-swingup'      :['Balance the pendulum by swinging it up with an under powered joint torque and then balance it',
                              'Using an under powered joint actuator, swing the pendulum up and balance it',
                              'You are given a pendulum with an underpowered joint actuator, swing the pendulum up and then balance it'],

    'hopper-hop'            :['Using an planar, one legged robot, stand up and hop forwards',
                              'Hop forwards with a planar one legged robot, by standing it up and then jump small hops',
                              'Stand the hopper up and then make it jump small jumps forward'],

    'hopper-stand'          :['Make the planar hopper robot stand up',
                              'Given a one legged planar robot, make it stand up and keep it standing',
                              'Stand this planar one legged hopper robot up and keep it standing'],

    'manipulator-insert_ball' :['Using a manipulator with a gripper, find the ball and insert it into the target area',
                                 'Bring the ball using the gripper attached to the manipulator',
                                'Find the ball and bring it to the target area using the manipulator'],

    'manipulator-bring_peg'  : ['Using a manipulator with a gripper, find the peg and hold it in the target area',
                                'Bring the peg with a manipulator with a gripper on the end',
                                'Find the peg in the play area and bring it with the gripper on the manipulator end']

}




