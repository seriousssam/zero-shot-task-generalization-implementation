#!/usr/bin/python
from envandnets import *
import random

grid_n = 6
init_reset_total_count = 1
blockSide = 32
numCellTypes = 8
episode_max_length = 50
init_reset_total_count = 60000
gpu = True
n = 5 #(number of tests per teacher during testing, 100 is standard 5 is just a quick check)
justprintsummary = False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                     Task representations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
imnames = ["empty",
"agent", #index 0 in our tensor
"greenball", #transformable0
#"greenguy", #transformable1
"blueball", #transformed0
#"yellowguy", #transformed1
"tv", #untransformable
"heart", #untransformable
"enemy",
"block",
"water"]

#change values depending on imnames
transformable_0_idx = 1
transformed_last_idx = 2
untransformable_0_idx = 3
untransformable_last_idx = 4

numTransformable = int((transformed_last_idx - transformable_0_idx + 1)/2)
numUntransformable = untransformable_last_idx - untransformable_0_idx + 1
enemy_idx = untransformable_last_idx + 1
block_idx = untransformable_last_idx + 2
water_idx = untransformable_last_idx + 3

#1s are the verbs, 2s are the objects
subtask_1s = ["Visited ", "Picked up ", "Transformed "] #, "Picked up all", "Transformed all"] #will deal with "all" later
subtask_2s = imnames[transformable_0_idx+1:untransformable_last_idx+2]
subtask_1s_num = len(subtask_1s)
subtask_2s_num = len(subtask_2s)
numTeachers = subtask_1s_num*subtask_2s_num

#takes a task in any format and converts it to pair format
def taskToPair(taskinput):
  if(isinstance(taskinput, str)):
    for i in range(subtask_1s_num):
      for j in range(subtask_2s_num):
        if(subtask_1s[i]+subtask_2s[j] == taskinput):
          return [i, j]
    raise ValueError('Task string input does not match any existing verb x object combo')
  elif(type(taskinput) is list):
    if(len(taskinput) == 2):
      return taskinput
    elif(len(taskinput == subtask_1s_num + subtask_2s_num)):
      i = -1
      for x in range(len(taskinput)):
        if(x==1):
          if(i==-1):
            i = x
          else:
            return [i, x-subtask_1s_num] #return our pair (first and second one)
        elif(x!=0):
          raise ValueError('1-hot encoding is badly formatted; should be just 1s and 0s') 
    else:
      raise ValueError('Task list input has an unexpected length \
      (should be 2 or subtask_1s_num + subtask_2s_num)')
  elif(type(taskinput) is int):
    if(taskinput >= numTeachers):
        ValueError('Task int input does not match any existing verb x object combo')
    else:
        sub2 = taskinput % subtask_2s_num
        sub1 = int((taskinput - sub2)/subtask_2s_num)
        return [sub1, sub2]
  else:
    raise ValueError('Task input is not a string or a list or an int')

#takes a task pair and turns it into a one hot encoding
def taskPairToOneHot(taskpair):
  out = [0] * (subtask_1s_num + subtask_2s_num)
  out[taskpair[0]] = 1
  out[subtask_1s_num + taskpair[1]] = 1
  return out

#takes a pair task and turns it into a task string
def taskPairToString(taskpair):
  return subtask_1s[taskpair[0]] + subtask_2s[taskpair[1]]

def taskPairToInt(taskpair):
  return subtask_2s_num*(taskpair[0]) + taskpair[1]  

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                        Utils
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def testOneTeacher(taskInt, teacher, teacherIsMulti = False): #can also take any task format
    taskString = taskPairToString(taskToPair(taskInt))
    
    avgNumSteps = []
    printingresults = False
    makeimages = False #make an image for each step
    #justprintsummary = True #only print numsteps and cumreward @ the end
    makegif = False

    numSolved = 0

    env = MazeWorld(grid_n,blockSide,numCellTypes, [taskString], episode_max_length, \
                        init_reset_total_count, gpu=gpu, changegrid=True)

    mode = "samtest"
    onehottask = torch.Tensor(taskPairToOneHot(taskToPair(taskInt)))
    onehottask = onehottask.cuda()
    
    for x in range(n):

        state = env.reset()
        submode = mode+"_" + str(x) + "_"

        if(makeimages or makegif):
            for fname in os.listdir('./'):
                if fname.startswith(submode) and fname.endswith(".jpg"):
                    softremove(os.path.join('.', fname))
        env.render(submode+ "0", 0, "none", 0, "Start", 0)

        if(logging):
            envlog = torch.zeros((episode_max_length+3,env.numCellTypes,env.grid_n,env.grid_n))
            envlog[0] = env.t
            actRewDoneLog = []

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                          Run an episode
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        done = False
        i = 1
        cumreward = 0

        while not done:
            if(teacherIsMulti):
                logit, _, _ = teacher((state.unsqueeze(0), onehottask)) #this will change later to accomodate value func
            else:
                logit, _ = teacher(state.unsqueeze(0))
            prob = F.softmax(logit, dim=1)
            action = prob.multinomial(1).data 

            state, reward, done, taskupdate = env.step(int(action[0]))
            renderfilename = submode + str(i)

            if(logging):
                envlog[i] = env.t
                actRewDoneLog.append([action, reward, done])

            actText = actionNumToText(int(action[0]))

            if(printingresults):
                print(str(i) + ": Action was " + actText +       ", reward is " + str(reward) + ", taskupdate is "       + taskupdate + ".")

            if(makeimages or makegif):
                env.render(renderfilename, i, int(action[0]), reward, taskupdate, cumreward)

            i += 1
            cumreward += reward

        if(reward>0):
            numSolved += 1
        elif(justprintsummary):
        #else:
            print("Attention: Teacher " + str(taskInt) + ", Episode " + str(x) + " didn't solve. See "\
                  + submode + ".gif for episode") 

        if(printingresults or justprintsummary):
            print("Teacher " + str(taskInt) + ", Episode " + str(x) + ": Env terminated after " +\
                str(i-1) + " steps, cumreward = " + str(cumreward))

        avgNumSteps.append(i-1)

        if(logging):
            softremove('envlog.tensor')
            torch.save(envlog, 'envlog.tensor')

        if(makegif):
            #code from Alex Buzunov at https://goo.gl/g2G9c6
            duration = 1
            filenames = sorted(filter(os.path.isfile, [x for x in os.listdir()   if (x.endswith(".jpg") and x.startswith(submode))]), key=lambda   p: os.path.exists(p) and os.stat(p).st_mtime or   time.mktime(datetime.now().timetuple()))

            #make an image that says "START"
            result = env.makeTextImage("START")
            result.save(submode + "Start.jpg")

            #make an image that says "END"
            result = env.makeTextImage("END")
            result.save(submode + "End.jpg")
            filenames.append(submode+"End.jpg")

            images = []
            images.append(imageio.imread(submode + "Start.jpg"))
            for filename in filenames:
                images.append(imageio.imread(filename))

            #output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
            output_file = submode + '.gif'
            imageio.mimsave(output_file, images, duration=duration)

            #EXPORT EPISODE TO DRIVE
            uploadIndividualImages = False #otherwise just upload gif
            if(uploadIndividualImages and goog):
                for filename in filenames:
                    f = drive.CreateFile()
                    f.SetContentFile(filename) #add the contents of your local file to the remove file
                    f.Upload() #create your remote file

            if(goog):
                f = drive.CreateFile()
                f.SetContentFile(mode+'.gif')
                f.Upload()

            #get rid of all the jpgs that start with submode
            if(makeimages or makegif):
                for fname in os.listdir('./'):
                    if fname.startswith(submode) and fname.endswith(".jpg"):
                        softremove(os.path.join('.', fname))

    if(justprintsummary):
        print("Thanks for playing with the " + taskString + "teacher!")
        print("numSolved is " + str(numSolved) + " out of " + str(n))
        print("Average number of steps is " + str(sum(avgNumSteps)/len(avgNumSteps)))

    return taskString, numSolved, n, avgNumSteps

def loadmodel(loadedModelPath):  
    try:
        model = teacherNetwork(grid_depth, act_dim)
        loadedDict = torch.load(loadedModelPath)
        model.load_state_dict(loadedDict)
        return model
    except OSError:
        #print("Errorcheck")
        return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                           Teachers data structure
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Teachers(object):
    def __init__(self, teachersDir, loadifpresent):
        self.loadifpresent = loadifpresent
        prefix = "teacher_"
        self.numTeachers = numTeachers
        
        self.arr = [None for x in range(self.numTeachers)]
        
        for tid in range(self.numTeachers):
            loadedModelPath = teachersDir + prefix + str(tid) + ".pt"
            #print(loadedModelPath)
            
            model = loadmodel(loadedModelPath)
            found = model != None
            
            if(self.loadifpresent[tid]!=1):
                model = None
            
            if(model!=None):
                model.cuda()
                print(str(tid) + " loaded")
            else:
                if(found):
                    print(str(tid) + " found but didn't load")
                else:
                    print(str(tid) + " not found, didn't load")
            
            self.arr[tid] = model
    
    def getTeacher(self, tid):
        return self.arr[tid]
    
    def testTask(self, taskInt):
        teacher = self.arr[taskInt]
        if(teacher == None):
            return taskPairToString(taskToPair(taskInt))
        else:
            return testOneTeacher(taskInt, teacher)
    
    def train(self, taskInts, ptask, savePath):
        self.arr
        self.numTeachers
        self.isLoadedArr = [0 for x in range(self.numTeachers)]
        for i in range(self.numTeachers):
            if(self.arr[i] != None):
                self.isLoadedArr[i] = 1
        
        self.numLoadedTeachers = sum(self.isLoadedArr)
        print("Training on " + str(len(taskInts)) + " teachers:")
        print("tid are: " + str(taskInts))#str([0,1,2]))
        
        if(gpu):
            ptask.cuda()
            
        trainer = Trainer(teachers, ptask, taskInts)
        trainer.train(savePath)
        #trainer.test()
        
        print()
        return trainer

    def testAllTasks(self, numeps = 100):
        resultsString = ""
        br = "\n"

        for taskInt in range(self.numTeachers):
            x = self.testTask(taskInt)
            #testOneTeacher(taskInt, teacher, teacherIsMulti = False)

            if(isinstance(x, str)):
                resultsString += "No teacher found for task \"" + x + "\"" + br
            else:
                taskString, numSolved, n, avgNumSteps = x
                resultsString += "For task \"" + taskString + ":\""
                resultsString += "numSolved is " + str(numSolved) + " out of " + str(n) + ", "
                resultsString += "average number of steps is " + \
                    str(sum(avgNumSteps)/len(avgNumSteps)) + br

        print(resultsString)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                PARAMETERIZED TASK
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Ptask(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Ptask, self).__init__()
        shared_linear_numhidden = 800
        embedding_size = grid_depth*(grid_depth+1)
        
        #self.conv0 = nn.Conv2d(num_inputs, num_inputs*4, 3)
        self.conv1 = nn.Conv2d(num_inputs, 256, 2)
        self.conv2 = nn.Conv2d(256, 128, 1)
        
        self.phi_1 = nn.Linear(subtask_1s_num,embedding_size)
        self.phi_2 = nn.Linear(subtask_2s_num,embedding_size)
        
        self.shared_linear1 = nn.Linear(3200 + embedding_size, 1600) #adding task embedding here
        self.shared_linear2 = nn.Linear(1600, shared_linear_numhidden)

        self.actor_linear = nn.Linear(shared_linear_numhidden, num_outputs)
        self.critic_linear = nn.Linear(shared_linear_numhidden, 1)
        self.term = nn.Linear(shared_linear_numhidden, 1)
        
        self.apply(weights_init)
        
        self.t_dis = 5
        self.t_diff = 5
        
        self.dis_p1 = 1
        self.diff_p2 = 1
        
        self.oneHotVecs = []
        for tid in range(numTeachers):
            x = torch.Tensor(taskPairToOneHot(taskToPair(tid)))
            x = x.cuda()
            self.oneHotVecs.append(x)
    
    def fwd_embedding(self, taskOneHot):
        taskOneHot_1 = taskOneHot[0:subtask_1s_num]
        taskOneHot_2 = taskOneHot[subtask_1s_num:]
        
        phi1out = self.phi_1(taskOneHot_1)
        phi2out = self.phi_2(taskOneHot_2)
        phiOut = F.relu(torch.mul(phi1out, phi2out))
        
        return phiOut
    
    def calc_analogy_loss(self, G_sim, G_diff, G_dis):
        oneHotVecs = self.oneHotVecs
        
        taskEmbeddings = [self.fwd_embedding(oneHotVecs[tid]) for tid  in range(numTeachers)]
        
        L_sim = 0
        for quadruplet in G_sim:
            a = taskEmbeddings[quadruplet[0]]
            b = taskEmbeddings[quadruplet[1]]
            c = taskEmbeddings[quadruplet[2]]
            d = taskEmbeddings[quadruplet[3]]
            
            L_sim += torch.norm(a-b-c+d)**2
        
        L_diff = 0
        for pair in G_diff:
            a = taskEmbeddings[pair[0]]
            b = taskEmbeddings[pair[1]]
            L_diff += max(self.t_diff - torch.norm(a-b), 0)**2
        
        L_dis = 0
        for quadruplet in G_dis:
            a = taskEmbeddings[quadruplet[0]]
            b = taskEmbeddings[quadruplet[1]]
            c = taskEmbeddings[quadruplet[2]]
            d = taskEmbeddings[quadruplet[3]]
            L_dis += max(self.t_dis - torch.norm(a-b-c+d),0)**2

        return L_sim + L_dis*self.dis_p1 + L_diff*self.diff_p2 
           
        
    def forward(self, inputs):
        x = inputs[0]
        taskOneHot = inputs[1]
        taskOneHot_1 = taskOneHot[0:subtask_1s_num]
        taskOneHot_2 = taskOneHot[subtask_1s_num:]
        
        phi1out = self.phi_1(taskOneHot_1)
        phi2out = self.phi_2(taskOneHot_2)
        phiOut = F.relu(torch.mul(phi1out, phi2out))
        
        embedding = phiOut[:-grid_depth]
        bias = phiOut[-grid_depth:]
        
        embedding = embedding.view(grid_depth, grid_depth, 1, 1)

        x = F.relu(F.conv2d(x, embedding, bias=bias))
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        x = torch.cat((x[0], phiOut), 0)
        x = x.unsqueeze(0)
        
        x = F.relu(self.shared_linear1(x))
        x = F.relu(self.shared_linear2(x))
        
        return self.actor_linear(x), F.sigmoid(self.term(x)), self.critic_linear(x)

'''
channels_for_embedding = 32
class Ptask(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(Ptask, self).__init__()
        shared_linear_numhidden = 400
        embedding_size = channels_for_embedding*16 + channels_for_embedding
        
        #self.conv0 = nn.Conv2d(num_inputs, num_inputs*4, 3)
        self.conv1 = nn.Conv2d(num_inputs, 16, 1)
        #self.conv2 = nn.Conv2d(64, 64, 1)
       
        self.phi = nn.Linear(subtask_1s_num + subtask_2s_num,embedding_size)
        #self.phi_1 = nn.Linear(subtask_1s_num,embedding_size)
        #self.phi_2 = nn.Linear(subtask_2s_num,embedding_size)
        
        self.shared_linear1 = nn.Linear(1152, 800) #adding task embedding here
        self.shared_linear2 = nn.Linear(800, shared_linear_numhidden)

        self.actor_linear = nn.Linear(shared_linear_numhidden, num_outputs)
        self.critic_linear = nn.Linear(shared_linear_numhidden, 1)
        self.term = nn.Linear(shared_linear_numhidden, 1)
        
        self.apply(weights_init)

    def forward(self, inputs):
        x = inputs[0]
        taskOneHot = inputs[1]
        
        #taskOneHot_1 = taskOneHot[0:subtask_1s_num]
        #taskOneHot_2 = taskOneHot[subtask_1s_num:]
        
        #phi1out = self.phi_1(taskOneHot_1)
        #phi2out = self.phi_2(taskOneHot_2)
        #phiOut = torch.mul(phi1out, phi2out)
        
        phiOut = self.phi(taskOneHot)
        
        embedding = phiOut[:-channels_for_embedding]
        bias = phiOut[-channels_for_embedding:]
        
        embedding = embedding.view(channels_for_embedding, 16, 1, 1)
        
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(F.conv2d(x, embedding, bias=bias))
        
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #b = 3/0
        
        x = x.view(x.size(0), -1)
                
        x = F.relu(self.shared_linear1(x))
        x = F.relu(self.shared_linear2(x))
        
        return self.actor_linear(x), F.sigmoid(self.term(x)), self.critic_linear(x)

'''

class Trainer():
    def __init__(self, teachers, ptask, teacherInts):
        self.teachers = teachers
        self.teacherInts = teacherInts
        self.ptask = ptask
        self.envs = [None for x in range(numTeachers)]
        
        #replay buffer stuff
        self.buffer = [[] for x in range(numTeachers)]
        self.bufferMaxLength = 1000 #samchange
        self.buffer_indices = [0 for x in range(numTeachers)]
        
        for tid in teacherInts:
            tnet = teachers.getTeacher(tid)
            taskString = taskPairToString(taskToPair(tid))
            print(taskString)
            self.envs[tid] = MazeWorld(grid_n,blockSide,numCellTypes, [taskString], episode_max_length, \
                    init_reset_total_count, gpu=gpu, changegrid=True)
            if(tnet==None):
                raise ValueError("Teacher " + str(tid) + " not loaded!")
            elif(gpu):
                teachers.getTeacher(tid).cuda()
    
    def initializeBuff(self, temperature, init_size=1000):
        #print("Initializing buffer for policy distillation")
        while (len(self.buffer[self.teacherInts[0]]) < init_size):
            for tid in self.teacherInts:
                env = self.envs[tid]
                state = env.reset()
                done = False
                tnet = teachers.getTeacher(tid)
                    
                while not done:
                    onehottask = torch.Tensor(taskPairToOneHot(taskToPair(tid)))
                    onehottask = onehottask.cuda()

                    state_ptask = state.unsqueeze(0)
                    state_tnet = state.unsqueeze(0)

                    ptask_logit, pterm, _ = self.ptask((state_ptask, onehottask)) #NO VALUE HERE!
                    tnet_logit, _ = tnet(state_tnet)

                    tnet_logit = tnet_logit.detach()
                    ptask_logit = ptask_logit.detach()
                    
                    tnet_logit = tnet_logit/temperature

                    ptask_prob = F.softmax(ptask_logit, dim=1)

                    action = ptask_prob.multinomial(1).data

                    state, reward, done, taskupdate = env.step(int(action[0]))

                    if(done and reward>0):
                        terminated = torch.Tensor([[1]])
                    else:
                        terminated = torch.Tensor([[0]])

                    #add to buffer
                    self.addSample([state, tnet_logit, terminated], tid)
        
        #print("Done initializing buffer!")
        
    #each sample is a state, logit_tnet, terminated (1 or 0) triplet
    def addSample(self, slt, tid):
        teacher_buff = self.buffer[tid]
        if(len(teacher_buff)<self.bufferMaxLength):
            teacher_buff.append(slt)
        else:
            teacher_buff[self.buffer_indices[tid]] = slt
            self.buffer_indices[tid] = (self.buffer_indices[tid] + 1 ) % self.bufferMaxLength
    
    #batch_size is the number of samples extracted for EACH teacher policy
    def getSamples(self, batch_size):
        sampleBuff = [[] for x in range(numTeachers)]
        for tid in range(numTeachers):
            teacher_buff = self.buffer[tid]
            if(len(teacher_buff) != 0):
                sample = random.sample(teacher_buff, batch_size)
                sampleBuff[tid] = sample
        return sampleBuff
    
    def train(self, savepath, numeps=50, numiters=41): # 5 and 100 normally
        optimizer = torch.optim.RMSprop(self.ptask.parameters(), lr=.0000025, alpha=0.97)
        criterion = nn.KLDivLoss()
        termCriterion = nn.BCELoss()
        alpha = .0001
        term_multiplier = 3
        multi_multiplier = 1
        temperature = 1 #.01#.01 #to sharpen the teacher!
        batch_size = 8 ###32
        
        self.initializeBuff(temperature, init_size=512)
        
        for itr in range(numiters):
            avgrewards = [0 for x in range(len(self.teacherInts))]
            avgLen = [0 for x in range(len(self.teacherInts))]
            numTerminated = [0 for x in range(len(self.teacherInts))]
            termMistakes_actuallyTerminated = [0 for x in range(len(self.teacherInts))]
            termMistakes_didNotTerminate = [0 for x in range(len(self.teacherInts))]
            num_didNotTerminate = [0 for x in range(len(self.teacherInts))]
            
            if(itr % 10 == 0 and itr > 0):
                netName = "ptask_" + str(itr) + ".pt"
                state_to_save = self.ptask.state_dict()
                torch.save(state_to_save, savepath + netName)
            
            for epnum in range(numeps):
                if(epnum%10 == 0):
                    print("episode " + str(epnum) + " running")
                
                ###total_loss = 0
                
                teachersPermutation = random.sample(self.teacherInts, len(self.teacherInts))
                ###i = 0
                for tid in teachersPermutation: ###self.teacherInts:
                    i = teachersPermutation.index(tid)
                    #print("Teacher " + str(tid) + " learning")
                    cumreward = 0
                    env = self.envs[tid]
                    tnet = teachers.getTeacher(tid)
                    
                    total_loss = 0
                
                    #RUN EPISODE
                    oldstate = env.reset()
                    done = False
                    numsteps = 0
                    
                    while not done:
                        onehottask = torch.Tensor(taskPairToOneHot(taskToPair(tid)))
                        onehottask = onehottask.cuda()
                        
                        state_ptask = Variable(oldstate.unsqueeze(0), requires_grad = True)
                        state_tnet = oldstate.unsqueeze(0)
                        
                        ptask_logit, pterm, _ = self.ptask((state_ptask, onehottask)) #NO VALUE HERE!
                        tnet_logit, _ = tnet(state_tnet)
                        
                        tnet_logit = tnet_logit.detach()
                        ptask_logit = ptask_logit.detach()
                        tnet_logit = tnet_logit/temperature
                        tnet_prob = F.softmax(tnet_logit, dim=1)

                        ptask_prob = F.softmax(ptask_logit, dim=1)
                        
                        action = ptask_prob.multinomial(1).data 
                        #tnetAction = tnet_prob.multinomial(1).data 
                        #tnetAction = int(torch.max(tnet_prob[0],0)[1])

                        state, reward, done, taskupdate = env.step(int(action[0]))
                        
                        if(reward>0 and done):
                            terminated = torch.Tensor([[1]])
                            multi_multiplier = term_multiplier
                            numTerminated[i] += 1
                            if(pterm[0] < .5):
                                termMistakes_actuallyTerminated[i] += 1
                        else:
                            multi_multiplier = 1
                            terminated = torch.Tensor([[0]])
                            if(pterm[0] > .5):
                                termMistakes_didNotTerminate[i] += 1
                            num_didNotTerminate[i] += 1
                            
                        terminated = terminated.cuda()

                        #add to buffer
                        self.addSample([oldstate, tnet_logit, terminated], tid)
                        
                        if(done): #train at end of episode
                            #train on buffer
                            total_loss = 0
                            sampleBuff = self.getSamples(batch_size)
                            for slt in sampleBuff[tid]:
                                s, l, t = slt
                                s = s.unsqueeze(0)

                                ptask_logit, pterm, _ = self.ptask((s, onehottask))
                                ptask_prob = F.softmax(ptask_logit, dim=1)

                                total_loss += criterion(ptask_prob, l)
                                total_loss += alpha * termCriterion(pterm, terminated) * multi_multiplier
                                #total_loss += criterion(ptask_prob[0][tnetAction], tnet_logit[0][tnetAction])
                                #total_loss += criterion(ptask_prob, tnet_logit)
                                
                        cumreward += reward
                        numsteps += 1
                        oldstate = state
                        
                        
                    self.ptask.zero_grad()
                    (total_loss).backward()
                    optimizer.step()
                    
                   
                    avgrewards[i] += cumreward
                    avgLen[i] += numsteps
                    ###i += 1
            
            ###self.ptask.zero_grad()
            ###(total_loss).backward()
            ###optimizer.step()
            
            avgrewards = [x/numeps for x in avgrewards]
            avgLen = [x/numeps for x in avgLen]
            
            avgavgRewards = round(sum(avgrewards)/len(avgrewards),2)
            avgavgLen = round(sum(avgLen)/len(avgLen),2)
            
            avgDidNotTerminate = [round(termMistakes_didNotTerminate[k]/num_didNotTerminate[k],2) for k in len(termMistakes_didNotTerminate)]
            
            print("For iteration " + str(itr) + ", avg reward is " + str(avgrewards) + " or " + str(avgavgRewards))
            print("Loss is " + str(int(total_loss)))
            print("Avg epLen is " + str(avgLen) + " or " + str(avgavgLen))
            print("Num completed is " + str(numTerminated) + " out of " + str(numeps) + " (total = " + str(sum(numTerminated)) + ")" )
            print("When ep terminated, numMistakes for pterm was " + str(termMistakes_actuallyTerminated))
            print("When ep did not terminate, pctMistakes for pterm was " + str(avgDidNotTerminate))
            print("~~~~~~")
            
    def testTasks(self, taskInts):
        resultsString = ""
        br = "\n"
        
        for taskInt in taskInts:
            x = testOneTeacher(taskInt, self.ptask, teacherIsMulti = True)
            
            taskString, numSolved, n, avgNumSteps = x
            
            resultsString += "For task \"" + taskString + ":\""
            resultsString += "numSolved is " + str(numSolved) + " out of " + str(n) + ", "
            resultsString += "average number of steps is " + \
                str(sum(avgNumSteps)/len(avgNumSteps)) + br

            #print(resultsString)
        return resultsString

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 META CONTROLLER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MetaController(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(MetaController, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 2)   # (10-4) / 1 + 1
  
        #self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 1)  # (7 + 2 -5 )/ 2 + 1
        self.conv3 = nn.Conv2d(32, 32, 1)  # (7 + 2 -5 )/ 2 + 1
        #self.maxp2 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(640,256,1)
        self.critic_linear = nn.Linear(256, numberofsubtask)
        self.actor_linear = nn.Linear(256, 13)

        self.phi_update = nn.Linear(256+800,1)
        
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()
        self.apply(weights_init)
        
    def forward(self, inputs):
        s_t, h_t_1, p_t_1, r_t_1, g_t_1 = inputs  #inputs is embedding of gt
        x = F.relu(self.conv1(s_t))
        x = F.relu((self.conv2(s_t)))
        x = F.relu((self.conv3(s_t)))
        
        c_t = F.sigmoid(self.phi_update(torch.cat((s_t.view(s_t.size(0)),h_t_t1),1)))
        x = x.view(x.size(0), -1)
        h_t_hat = self.lstm(x, (h_t_1))
        I_t = F.softmax(phi_shift(ht))
        p_t_hat = I_t * p_t_1
        r_t_hat  = M * p_t
        
        p_t, r_t, h_t = c_t * [p_t_hat, r_t_hat, h_t_hat] + (1 - c_t) * [p_t_1, r_t_1,h_t_1] #doing soft update
        
        g_t  = c_t * exp(phi_goal(h_t,r_t)) + (1 - c_t) * g_t_1
        
        return self.critic_linear(x), self.actor_linear(x), h_t, p_t, r_t, g_t
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                 ACTOR CRITIC FINE-TUNING
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ptaskActorCritic(rank, taskInts, gamma, tau, shared_ptask, optimizer, num_iter, num_steps=1000, report_iter=500):
    global numIterSoFar
    global numIterToNoEntropyLoss
    log_dir = 'models/dpt/'
    zeta_analogy = 2
    br = "\n"
    
    #ptaskActorCritic # BE CAREFUL!
    alpha = .0001#.0001
    termCriterion = nn.BCELoss()

    if(rank>-1):
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)

    lossLog = []

    myptask = Ptask(8, 13)

    if(rank>-1):
        myptask.cuda()

    numDone = 0
    totalDone = 0
    ep_lens = []
    print("Process " + str(rank) + " starting training" + ", time is ", end=" ")
    print(datetime.datetime.now().strftime('%H-%M-%S'))
    
    num_eps_per_iter = 1 #len(taskInts)
    bestAvgLen = 1000
    bestAvgLen_score = 0
    
    onehottasks = [[] for x in range(numTeachers)]
    for i in range(numTeachers):
        onehottasks[i] = torch.Tensor(taskPairToOneHot(taskToPair(i)))
        onehottasks[i] = onehottasks[i].cuda()

    envs = [None for x in range(numTeachers)]
    for tid in range(numTeachers):
        taskString = taskPairToString(taskToPair(tid))
        envs[tid] = MazeWorld(grid_n,blockSide,numCellTypes, [taskString], episode_max_length, \
            init_reset_total_count, gpu=gpu, changegrid=True)
    
        
    num_actualTerminated = 0 
    num_nonTerminal = 0 
    num_mistakes_terminated = 0
    num_mistakes_nonTerminal = 0
    
    for iter in range(num_iter):
        
        if(iter%report_iter==0 and iter > 0):
            
            report = "==============================================================================" + br
            
            report += "Process " + str(rank) +  " REPORTING TIME!! ~~~~~~~~~~~!@#&(*&(*(&@!*&(*@!&#@! REPORTING PAY ATTENTION!" + br
            tsknt = random.choice([2, 5])
            x = testOneTeacher(tsknt, myptask, teacherIsMulti = True)
            
            taskString, numSolved, n, avgNumSteps = x
            
            try:
                report += "Testing on unseen task to check analogy: for task \"" + taskString + ":\" aka " + str(tsknt) + br
            except ValueError:
                report += "ValueErr just happened!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" + br
            report += "numSolved is " + str(numSolved) + " out of " + str(n) + ", " + br
            report += "average number of steps is " + str(sum(avgNumSteps)/len(avgNumSteps)) + br
            
            
            report += "Now reporting on pterm performance:" + br
            if(num_actualTerminated > 0):
                report += "On terminal states, performance is " + str(1-round(num_mistakes_terminated/num_actualTerminated,2)) + br
            if(num_nonTerminal > 0):
                report += "On nonterminal states, performance is " + str(1-round(num_mistakes_nonTerminal/num_nonTerminal,2)) + br
            report += "END REPORTING================================================================" + br
            
            print(report)
            
            num_actualTerminated = 0 
            num_nonTerminal = 0 
            num_mistakes_terminated = 0
            num_mistakes_nonTerminal = 0
            
        
        #print("iter " + str(iter))
        annealing = max(0, 1-numIterSoFar/numIterToNoEntropyLoss)
        
        #if(iter%numiters_between_reset_total_count_decrements==0 and iter > 0 and env.reset_total_count>1):
            #env.decrement_reset_total_count() if I had time to build it
            #env.reset_total_count -= 1

        if(iter%100==0 and iter > 0):
          if(len(ep_lens)>0):
            avgLen = sum(ep_lens)/len(ep_lens)
            if(bestAvgLen>avgLen):
                bestAvgLen = avgLen
                bestAvgLen_iter = int(iter)
                bestAvgLen_score = str(numDone) + " out of " + str(totalDone)
                state_to_save = myptask.state_dict()
                torch.save(state_to_save, '{0}{1}{2}.pt'.format(
                        log_dir, "bestmodel_", rank))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Process " + str(rank) + " Training; current iter: " + str(iter) + ", time is ", end=" ")
            print(datetime.datetime.now().strftime('%H-%M-%S'))
            print("Number of times it's done before maxsteps is " + str(numDone) + " out of " + str(totalDone))
            print("Average episode length is " + str(avgLen))
            print("So far bestAvgLen is " + str(bestAvgLen) + " achieved at iter " + \
                  str(bestAvgLen_iter) + " with " + bestAvgLen_score)
            ep_lens = []
            numDone = 0
            totalDone = 0

        myptask.load_state_dict(shared_ptask.state_dict())
        
        total_loss = 0
        pterm_loss = 0
        
        for x in range(num_eps_per_iter):
            PTSKvalues = []
            PTSKrewards = []
            PTSKlog_probs = []
            PTSKentropies = []
            
            tid = random.choice(taskInts)
            env = envs[tid]
            state = env.reset()
            
            
            for step in range(num_steps):
                logit, pterm, value = myptask((Variable(state.unsqueeze(0)), Variable(onehottasks[tid])))
                prob = F.softmax(logit, dim=1)
                log_prob = F.log_softmax(logit, dim=1)
                entropy = -(log_prob * prob).sum(1)
                PTSKentropies.append(entropy)

                action = prob.multinomial(1).data
                log_prob2 = log_prob.gather(1, Variable(action))
                state, reward, done, taskupdate = env.step(action.cpu().numpy())

                PTSKvalues.append(value)
                PTSKlog_probs.append(log_prob2)
                PTSKrewards.append(reward)
                
                if(done and reward > 0):
                    terminated = torch.Tensor([[1]])
                    num_actualTerminated += 1
                    if(pterm[0] < .5):
                        num_mistakes_terminated += 1
                else:
                    terminated = torch.Tensor([[0]])
                    num_nonTerminal += 1
                    if(pterm[0] > .5):
                        num_mistakes_nonTerminal += 1
                terminated = terminated.cuda()
                
                pterm_loss += alpha * termCriterion(pterm, terminated)
              
                if done:
                    totalDone += 1
                    ep_lens.append(step)

                    if(PTSKrewards[-1]>0):
                      numDone += 1

                    break            

            if(rank>-1):
                PTSKvalues.append(torch.zeros(1, 1).cuda())
            else:
                PTSKvalues.append(torch.zeros(1, 1))

            policy_loss = 0
            value_loss = 0

            if(rank>-1):
                gae = torch.zeros(1, 1).cuda()
            else:
                gae = torch.zeros(1, 1)

            #GAE Calc
            for i in reversed(range(len(PTSKrewards))):
                R = PTSKrewards[i]
                advantage = R - PTSKvalues[i]
                value_loss = value_loss + advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = PTSKrewards[i] + gamma * PTSKvalues[i + 1].data - PTSKvalues[i].data

                gae = gae * gamma * tau + delta_t

                policy_loss = policy_loss - PTSKlog_probs[i]*Variable(gae) - annealing*.05*PTSKentropies[i]

            #Calculate loss and backprop
            analogy_loss = myptask.calc_analogy_loss(G_sim, G_diff, G_dis)
            
            if(iter%report_iter == 0 and iter > 0 ):
                print("Analogy loss is " + str(analogy_loss))

            total_loss = total_loss + policy_loss + 0.5 * value_loss + zeta_analogy*analogy_loss + pterm_loss
            lossLog.append(float(policy_loss + 0.5 * value_loss))
            #player.clear_actions()

        myptask.zero_grad()
        (total_loss).backward()
        ensure_shared_grads(myptask, shared_ptask)
        optimizer.step()
        
        numIterSoFar += 1

    #samget
    state_to_save = myptask.state_dict()
    torch.save(state_to_save, 'ptask_final' + str(rank) + '.pt')
    return lossLog

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       Task groups for analogy-making loss
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
trainingTasks = [0, 1, 3, 4, 6, 7, 8, 10]
G = range(numTeachers)

G_sim = [[7,4,3,0], [10,8,6,4]]
G_diff = []
G_dis = []

for i in reversed(range(len(trainingTasks))):
    for j in reversed(range(i)):
        G_diff.append([trainingTasks[i], trainingTasks[j]])

for i in reversed(range(len(trainingTasks))):
    for j in reversed(range(i)):
        for k in reversed(range(j)):
            for l in reversed(range(k)):
                a = trainingTasks[i]
                b = trainingTasks[j]
                c = trainingTasks[k]
                d = trainingTasks[l]
                
                if(not [a,b,c,d] in G_sim):
                    G_dis.append([a,b,c,d])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                        MAIN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
teachersDir = 'models/teachers/'

main = __name__  ==  '__main__'
loadifpresent = [1 for x in range(numTeachers)]
#loadifpresent[4] = 0 # only testing tasks 0, 1, 2
#loadifpresent[1] = 0
#loadifpresent[2] = 0



testTeachers = False
loadPtask = False
distillPtask = False
fineTunePtask = True
testPtask = True

savePath = 'models/ptasks/'
loadFile = 'models/dpt/bestmodel_0.pt'
#savePath + "ptask_final_refine.pt"

taskInts = trainingTasks

numWorkers = 4

if(main):
    print("Loading teachers...")
    teachers = Teachers(teachersDir, loadifpresent)
    print()
    
    if(testTeachers):
        print("Starting testing...")
        teachers.testAllTasks()

    ptask = Ptask(8, 13)
    
    if(loadPtask):
        print("Loading ptask...")
        bestDict = torch.load(loadFile)
        ptask.load_state_dict(bestDict)
    ptask.cuda()
        
    if(distillPtask):
        print("Starting training...")
        trainer = teachers.train(taskInts, ptask, savePath)
        state_to_save = ptask.state_dict()
        torch.save(state_to_save, savePath + "ptask_final.pt")
        
    if(fineTunePtask):
        mp.set_start_method('spawn')
        mp.freeze_support()
        
        if(gpu):
            torch.cuda.manual_seed(0)

        processes = []
        shared_ptask = ptask
        optimizer = SharedRMSprop(shared_ptask.parameters(), lr=.000001)

        for rank in range(0, numWorkers):
            print(rank)
            if(not gpu):
                rank = -1
            
            p = mp.Process(
                target = ptaskActorCritic, args=(rank, taskInts, gamma, tau, shared_ptask, optimizer, 2501))#16001
            p.start()
            processes.append(p)
            time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()
        
        state_to_save = ptask.state_dict()
        torch.save(state_to_save, savePath + "ptask_final_refine.pt")
        
    if(testPtask):
        print("Testing task...")
        allTasks = range(12)
        newtrainer = Trainer(teachers, ptask, taskInts)
        resultsString = newtrainer.testTasks(taskInts)
        print(resultsString)