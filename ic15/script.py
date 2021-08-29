#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import rrc_evaluation_funcs
import importlib
import time
import sys
import os
import threading
import math
import pdb
import numpy as np
import Polygon as plg
import cv2

from multiprocessing import Process,Queue

def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """    
    return {
            'Polygon':'plg',
            'numpy':'np'
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
                'IOU_CONSTRAINT' :0.5,
                'AREA_PRECISION_CONSTRAINT' :0.5,
                'Test_Image_ID': 'img_([0-9]+).jpg',
                'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
                'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',
                'LTRB':False, #LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
                'CRLF':False, # Lines are delimited by Windows CRLF format
                'CONFIDENCES':False, #Detections must include confidence value. AP will be calculated
                'PER_SAMPLE_RESULTS':True #Generate per sample results and produce data for visualization
            }

# 验证 数据格式正确性
def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])

    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
    
    #Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(k,gt[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True)

    #Validate format of results
    for k in subm:
        if (k in gt) == False :
            raise Exception("The sample %s not present in GT" %k)
        
        rrc_evaluation_funcs.validate_lines_in_file(k,subm[k],evaluationParams['CRLF'],evaluationParams['LTRB'],False,evaluationParams['CONFIDENCES'])

'''    
def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    #                    p['g'], p['s'], evalParams
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """    
    
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)    
    
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """        
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0])
        resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2])
        resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4])
        resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6])
        resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg.Polygon( pointMat)
    
    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax)
        resBoxes[0,7]=int(rect.ymax)

        pointMat = resBoxes[0].reshape([2,4]).T
        
        return plg.Polygon( pointMat)
    
    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points
        
    def get_union(pD,pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)
        
    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def funcCt(x):
        if x<=0.01:
            return 1
        else:
            return 1-x
    
    def get_text_intersection_over_union_recall(pD, pG):
        
        # Ct (cut): Area of ground truth that is not covered by detection bounding box.
        
        try:
            Ct = pG.area() - get_intersection(pD, pG)
            assert(Ct>=0 and Ct<=pG.area()), 'Invalid Ct value'
            assert(pG.area()>0), 'Invalid Gt'

            #     TIOU_recall =  A(Gi交Di)*f(Ct)
            return (get_intersection(pD, pG) * funcCt(Ct*1.0/pG.area())) / get_union(pD, pG)
        except Exception as e:
            return 0

    
    def funcOt(x):
        if x<=0.01:
            return 1
        else:
            return 1-x

    
    def get_text_intersection_over_union_precision(pD, pG, gtNum, gtPolys, gtDontCarePolsNum):
        
        #Ot: Outlier gt area
        
        Ot = 0
        try:
            inside_pG = pD & pG
            gt_union_inside_pD = None
            gt_union_inside_pD_and_pG = None
            count_initial = 0
            for i in range(len(gtPolys)):
                if i!= gtNum and gtNum not in gtDontCarePolsNum: # ignore don't care regions
                    if not get_intersection(pD, gtPolys[i]) == 0:
                        if count_initial == 0:
                            # initial 
                            gt_union_inside_pD = gtPolys[i]
                            gt_union_inside_pD_and_pG = inside_pG & gtPolys[i]
                            count_initial = 1
                            continue
                        gt_union_inside_pD = gt_union_inside_pD | gtPolys[i]
                        inside_pG_i = inside_pG & gtPolys[i]
                        gt_union_inside_pD_and_pG = gt_union_inside_pD_and_pG | inside_pG_i

            if not gt_union_inside_pD == None:
                pD_union_with_other_gt = pD & gt_union_inside_pD
                
                #
                Ot = pD_union_with_other_gt.area() - gt_union_inside_pD_and_pG.area()
                if Ot <=1.0e-10:
                    Ot = 0
            else:
                Ot = 0
            assert(Ot>=0 and Ot<=pD.area()+0.001), 'Invalid Ot value: '+str(Ot)+' '+str(pD.area())
            assert(pD.area()>0), 'Invalid pD: '+str(pD.area())

            return (get_intersection(pD, pG) * funcOt(Ot*1.0/pD.area())) / get_union(pD, pG)
        except Exception as e:
            print(e)
            return 0


    def get_intersection(pD,pG):
        
        pInt = pD & pG

        if len(pInt) == 0:
            return 0
        return pInt.area()

    
    def get_intersection_three(pD,pG,pGi):
        pInt = pD & pG
        pInt_3 = pInt & pGi
        if len(pInt_3) == 0:
            return 0
        return pInt_3.area()
    
    def compute_ap(confList, matchList,numGtCare):
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if numGtCare>0:
                AP /= numGtCare
            
        return AP
    
    perSampleMetrics = {}
    
    matchedSum = 0
    matchedSum_iou = 0 

    matchedSum_tiouGt = 0 
    matchedSum_tiouDt = 0 

    matchedSum_cutGt = 0 
    matchedSum_coverOtherGt = 0 
    
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    
    # key=id  value=label
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
   
    numGlobalCareGt = 0
    numGlobalCareDet = 0
    
    arrGlobalConfidences = []
    arrGlobalMatches = []

    totalNumGtPols = 0
    totalNumDetPols = 0
    
    fper_ = open('per_samle_result.txt', 'w')

    # import time 
    # st = time.time()
    for resFile in gt:
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0    
        
        detMatched = 0
        detMatched_iou = 0 
        detMatched_tiouGt = 0 
        detMatched_tiouDt = 0 
        detMatched_cutGt = 0 
        detMatched_coverOtherGt = 0 
        
        iouMat = np.empty([1,1])
        
        # Polygon : shape=[[x1,y1], [x2,y2],......]
        gtPols = []
        detPols = []
        # raw gt/pre decode point: shape= x1,y1,x2,y2,x3,y3,x4,y4
        gtPolPoints = []
        detPolPoints = []  
        
        #Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        #Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []
        
        pairs = []
        detMatchedNums = []
        
        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""
        
        # 按行 读取 gt label
        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:            
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                # 记录gt中为“###”的索引
                gtDontCarePolsNum.append( len(gtPols)-1 )
                
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")
        

        # 读取 gt 对应的 pre输出的结果
        if resFile in subm:
            
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 
            
            pointsList, confidencesList, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],False,evaluationParams['CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]
                
                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)                    
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum)>0 :
                    # 遍历 每一个 gt中为"###"的 框
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break
                                
            evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n")
            
            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols), len(detPols)]

                iouMat = np.empty(outputShape)

                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)

                # 一张图片标签里的 第i个的 框
                tiouRecallMat = np.empty(outputShape)  
                tiouPrecisionMat = np.empty(outputShape)  

                # 
                # tiouGtRectMat = np.zeros(len(gtPols), np.int8) 
                # tiouDetRectMat = np.zeros(len(detPols), np.int8) 

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]

                        # IOU_gt_i
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                        # TIOU_recall_gt_i
                        tiouRecallMat[gtNum,detNum] = get_text_intersection_over_union_recall(pD, pG)
                        # TIOU_precision_i
                        tiouPrecisionMat[gtNum,detNum] = get_text_intersection_over_union_precision(pD, pG, gtNum, gtPols, gtDontCarePolsNum)  

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:

                                # IOU
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1

                                # SIOU
                                detMatched += 1
                                detMatched_iou += iouMat[gtNum,detNum]

                                # TIOU
                                detMatched_tiouGt += tiouRecallMat[gtNum,detNum] 
                                detMatched_tiouDt += tiouPrecisionMat[gtNum,detNum]

                                if  iouMat[gtNum,detNum] != tiouRecallMat[gtNum,detNum]: 
                                    detMatched_cutGt +=1

                                if  iouMat[gtNum,detNum] != tiouPrecisionMat[gtNum,detNum]: 
                                    detMatched_coverOtherGt +=1

                                pairs.append({'gt':gtNum,'det':detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum :
                        #we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        # 除去 无效的框
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare >0 else float(1)
            sampleAP = precision
            iouRecall = float(1)
            iouPrecision = float(0) if numDetCare > 0 else float(1)
            tiouRecall = float(1) 
            tiouPrecision = float(0) if numDetCare >0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare==0 else float(detMatched) / numDetCare

            iouRecall = float(detMatched_iou) / numGtCare 
            iouPrecision = 0 if numDetCare==0 else float(detMatched_iou) / numDetCare 

            tiouRecall = float(detMatched_tiouGt) / numGtCare 
            tiouPrecision = 0 if numDetCare==0 else float(detMatched_tiouDt) / numDetCare 

            if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )

        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)
        tiouHmean = 0 if (tiouPrecision + tiouRecall)==0 else 2.0 * tiouPrecision * tiouRecall / (tiouPrecision + tiouRecall)     
        iouHmean = 0 if (iouPrecision + iouRecall)==0 else 2.0 * iouPrecision * iouRecall / (iouPrecision + iouRecall)     

        # SIOU
        matchedSum += detMatched
        matchedSum_iou += detMatched_iou 

        # TIOU
        matchedSum_tiouGt += detMatched_tiouGt 
        matchedSum_tiouDt += detMatched_tiouDt 

        matchedSum_cutGt += detMatched_cutGt 
        matchedSum_coverOtherGt += detMatched_coverOtherGt 

        # 一张图片中 有效框的数量
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        
        # 将结果保存为.txt
        if evaluationParams['PER_SAMPLE_RESULTS']:
            perSampleMetrics[resFile] = {
                                            'precision':precision,
                                            'recall':recall,
                                            'hmean':hmean,
                                            'iouPrecision':iouPrecision,
                                            'iouRecall':iouRecall,
                                            'iouHmean':iouHmean,
                                            'tiouPrecision':tiouPrecision,
                                            'tiouRecall':tiouRecall,
                                            'tiouHmean':tiouHmean,
                                            'pairs':pairs,
                                            'AP':sampleAP,
                                            'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
                                            'gtPolPoints':gtPolPoints,
                                            'detPolPoints':detPolPoints,
                                            'gtDontCare':gtDontCarePolsNum,
                                            'detDontCare':detDontCarePolsNum,
                                            'evaluationParams': evaluationParams,
                                            'evaluationLog': evaluationLog                                        
                                        }
        fper_.writelines(resFile+'\t"IoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})",\t"TIoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})".\n'.format(precision, recall, hmean, tiouPrecision, tiouRecall, tiouHmean))
        try:
            totalNumGtPols += len(gtPols) 
            totalNumDetPols += len(detPols)
        except Exception as e:
            raise e
    fper_.close()




    # Compute MAP and MAR
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    print('num_gt, num_det: ', numGlobalCareGt, totalNumDetPols)

    # iou
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    # siou
    methodRecall_iou = 0 if numGlobalCareGt == 0 else float(matchedSum_iou)/numGlobalCareGt 
    methodPrecision_iou = 0 if numGlobalCareDet == 0 else float(matchedSum_iou)/numGlobalCareDet 
    iouMethodHmean = 0 if methodRecall_iou + methodPrecision_iou==0 else 2* methodRecall_iou * methodPrecision_iou / (methodRecall_iou + methodPrecision_iou) 

    # tiou
    methodRecall_tiouGt = 0 if numGlobalCareGt == 0 else float(matchedSum_tiouGt)/numGlobalCareGt 
    methodPrecision_tiouDt = 0 if numGlobalCareDet == 0 else float(matchedSum_tiouDt)/numGlobalCareDet 
    tiouMethodHmean = 0 if methodRecall_tiouGt + methodPrecision_tiouDt==0 else 2* methodRecall_tiouGt * methodPrecision_tiouDt / (methodRecall_tiouGt + methodPrecision_tiouDt) 
    
    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean}
    iouMethodMetrics = {'iouPrecision':methodPrecision_iou, 'iouRecall':methodRecall_iou,'iouHmean': iouMethodHmean }
    tiouMethodMetrics = {'tiouPrecision':methodPrecision_tiouDt, 'tiouRecall':methodRecall_tiouGt,'tiouHmean': tiouMethodHmean }

    # et = time.time()
    # print("[INFO] Time is:{}".format(et - st))

    # print('matchedSum: ', matchedSum, 'matchedSum_cutGt: ', matchedSum_cutGt, 'cut_Rate: ', round(matchedSum_cutGt*1.0/matchedSum, 3), 'matchedSum_coverOtherGt: ', matchedSum_coverOtherGt, 'cover_Outlier_Rate: ', round(matchedSum_coverOtherGt*1.0/matchedSum, 3))
    print('Origin:')
    print("recall: ", round(methodRecall,3), "precision: ", round(methodPrecision,3), "hmean: ", round(methodHmean,3))
    print('SIoU-metric:')
    print("iouRecall:", round(methodRecall_iou,3), "iouPrecision:", round(methodPrecision_iou,3), "iouHmean:", round(iouMethodHmean,3))
    print('TIoU-metric:')
    print("tiouRecall:", round(methodRecall_tiouGt,3), "tiouPrecision:", round(methodPrecision_tiouDt,3), "tiouHmean:", round(tiouMethodHmean,3))

    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics, 'iouMethod': iouMethodMetrics, 'tiouMethod': tiouMethodMetrics}
    
    
    return resDict

'''

def evaluate_method(gtFilePath, submFilePath, evaluationParams, num):
    #                    p['g'], p['s'], evalParams
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """    
    
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)    
    
    # def polygon_from_points(points):
    #     """
    #     Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    #     """        
    #     resBoxes=np.empty([1,8],dtype='int32')
    #     resBoxes[0,0]=int(points[0])
    #     resBoxes[0,4]=int(points[1])
    #     resBoxes[0,1]=int(points[2])
    #     resBoxes[0,5]=int(points[3])
    #     resBoxes[0,2]=int(points[4])
    #     resBoxes[0,6]=int(points[5])
    #     resBoxes[0,3]=int(points[6])
    #     resBoxes[0,7]=int(points[7])
    #     pointMat = resBoxes[0].reshape([2,4]).T
    #     return plg.Polygon( pointMat)
    
    # def rectangle_to_polygon(rect):
    #     resBoxes=np.empty([1,8],dtype='int32')
    #     resBoxes[0,0]=int(rect.xmin)
    #     resBoxes[0,4]=int(rect.ymax)
    #     resBoxes[0,1]=int(rect.xmin)
    #     resBoxes[0,5]=int(rect.ymin)
    #     resBoxes[0,2]=int(rect.xmax)
    #     resBoxes[0,6]=int(rect.ymin)
    #     resBoxes[0,3]=int(rect.xmax)
    #     resBoxes[0,7]=int(rect.ymax)

    #     pointMat = resBoxes[0].reshape([2,4]).T
        
    #     return plg.Polygon( pointMat)
    
    # def rectangle_to_points(rect):
    #     points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
    #     return points
        
    # def get_union(pD,pG):
    #     areaA = pD.area()
    #     areaB = pG.area()
    #     return areaA + areaB - get_intersection(pD, pG)
        
    # def get_intersection_over_union(pD,pG):
    #     try:
    #         return get_intersection(pD, pG) / get_union(pD, pG)
    #     except:
    #         return 0

    # def funcCt(x):
    #     if x<=0.01:
    #         return 1
    #     else:
    #         return 1-x
    
    # def get_text_intersection_over_union_recall(pD, pG):
        
    #     # Ct (cut): Area of ground truth that is not covered by detection bounding box.
        
    #     try:
    #         Ct = pG.area() - get_intersection(pD, pG)
    #         assert(Ct>=0 and Ct<=pG.area()), 'Invalid Ct value'
    #         assert(pG.area()>0), 'Invalid Gt'

    #         #     TIOU_recall =  A(Gi交Di)*f(Ct)
    #         return (get_intersection(pD, pG) * funcCt(Ct*1.0/pG.area())) / get_union(pD, pG)
    #     except Exception as e:
    #         return 0

    
    # def funcOt(x):
    #     if x<=0.01:
    #         return 1
    #     else:
    #         return 1-x

    
    # def get_text_intersection_over_union_precision(pD, pG, gtNum, gtPolys, gtDontCarePolsNum):
        
    #     # Ot: Outlier gt area
        
    #     Ot = 0
    #     try:
    #         inside_pG = pD & pG
    #         gt_union_inside_pD = None
    #         gt_union_inside_pD_and_pG = None
    #         count_initial = 0
    #         for i in range(len(gtPolys)):
    #             if i!= gtNum and gtNum not in gtDontCarePolsNum: # ignore don't care regions
    #                 if not get_intersection(pD, gtPolys[i]) == 0:
    #                     if count_initial == 0:
    #                         # initial 
    #                         gt_union_inside_pD = gtPolys[i]
    #                         gt_union_inside_pD_and_pG = inside_pG & gtPolys[i]
    #                         count_initial = 1
    #                         continue
    #                     gt_union_inside_pD = gt_union_inside_pD | gtPolys[i]
    #                     inside_pG_i = inside_pG & gtPolys[i]
    #                     gt_union_inside_pD_and_pG = gt_union_inside_pD_and_pG | inside_pG_i

    #         if not gt_union_inside_pD == None:
    #             pD_union_with_other_gt = pD & gt_union_inside_pD
                
    #             #
    #             Ot = pD_union_with_other_gt.area() - gt_union_inside_pD_and_pG.area()
    #             if Ot <=1.0e-10:
    #                 Ot = 0
    #         else:
    #             Ot = 0
    #         assert(Ot>=0 and Ot<=pD.area()+0.001), 'Invalid Ot value: '+str(Ot)+' '+str(pD.area())
    #         assert(pD.area()>0), 'Invalid pD: '+str(pD.area())

    #         return (get_intersection(pD, pG) * funcOt(Ot*1.0/pD.area())) / get_union(pD, pG)
    #     except Exception as e:
    #         print(e)
    #         return 0


    # def get_intersection(pD,pG):
        
    #     pInt = pD & pG

    #     if len(pInt) == 0:
    #         return 0
    #     return pInt.area()

    
    # def get_intersection_three(pD,pG,pGi):
    #     pInt = pD & pG
    #     pInt_3 = pInt & pGi
    #     if len(pInt_3) == 0:
    #         return 0
    #     return pInt_3.area()
    
    # def compute_ap(confList, matchList,numGtCare):
    #     correct = 0
    #     AP = 0
    #     if len(confList)>0:
    #         confList = np.array(confList)
    #         matchList = np.array(matchList)
    #         sorted_ind = np.argsort(-confList)
    #         confList = confList[sorted_ind]
    #         matchList = matchList[sorted_ind]
    #         for n in range(len(confList)):
    #             match = matchList[n]
    #             if match:
    #                 correct += 1
    #                 AP += float(correct)/(n + 1)

    #         if numGtCare>0:
    #             AP /= numGtCare
            
    #     return AP
    
    # perSampleMetrics = {}
    
    # matchedSum = 0
    # matchedSum_iou = 0 

    # matchedSum_tiouGt = 0 
    # matchedSum_tiouDt = 0 

    # matchedSum_cutGt = 0 
    # matchedSum_coverOtherGt = 0 
    
    # Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    
    # # key=id  value=label
    # gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    # subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
    
    # numGlobalCareGt = 0
    # numGlobalCareDet = 0
    
    # arrGlobalConfidences = []
    # arrGlobalMatches = []

    # totalNumGtPols = 0
    # totalNumDetPols = 0


    # threads = []

    # # for t in range(0,3):
    # #     t= threading.Thread(target=read_file,args=("D:/zhihu/",x))
    # #     threads.append(t)
    
    # fper_ = open('per_samle_result.txt', 'w')

    # import time 
    # st = time.time()
    # # print("gt:", gt)

    # for resFile in gt:

    #     resFile = str(resFile)

    #     gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
    #     recall = 0
    #     precision = 0
    #     hmean = 0    
        
    #     detMatched = 0
    #     detMatched_iou = 0 
    #     detMatched_tiouGt = 0 
    #     detMatched_tiouDt = 0 
    #     detMatched_cutGt = 0 
    #     detMatched_coverOtherGt = 0 
        
    #     iouMat = np.empty([1,1])
        
    #     # Polygon : shape=[[x1,y1], [x2,y2],......]
    #     gtPols = []
    #     detPols = []
    #     # raw gt/pre decode point: shape= x1,y1,x2,y2,x3,y3,x4,y4
    #     gtPolPoints = []
    #     detPolPoints = []  
        
    #     #Array of Ground Truth Polygons' keys marked as don't Care
    #     gtDontCarePolsNum = []
    #     #Array of Detected Polygons' matched with a don't Care GT
    #     detDontCarePolsNum = []
        
    #     pairs = []
    #     detMatchedNums = []
        
    #     arrSampleConfidences = []
    #     arrSampleMatch = []
    #     sampleAP = 0

    #     evaluationLog = ""
        
    #     # 按行 读取 gt label
    #     pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        
    #     for n in range(len(pointsList)):
    #         points = pointsList[n]
    #         transcription = transcriptionsList[n]
    #         dontCare = transcription == "###"
    #         if evaluationParams['LTRB']:
                
    #             gtRect = Rectangle(*points)
    #             gtPol = rectangle_to_polygon(gtRect)
    #         else:            
    #             gtPol = polygon_from_points(points)
    #         gtPols.append(gtPol)
    #         gtPolPoints.append(points)
    #         if dontCare:
    #             # 记录gt中为“###”的索引
    #             gtDontCarePolsNum.append( len(gtPols)-1 )
                
    #     evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")
        

    #     # 读取 gt 对应的 pre输出的结果
    #     if resFile in subm:
            
    #         detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 
            
    #         pointsList, confidencesList, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],False,evaluationParams['CONFIDENCES'])
    #         for n in range(len(pointsList)):
    #             points = pointsList[n]
                
    #             if evaluationParams['LTRB']:
    #                 detRect = Rectangle(*points)
    #                 detPol = rectangle_to_polygon(detRect)
    #             else:
    #                 detPol = polygon_from_points(points)                    
    #             detPols.append(detPol)
    #             detPolPoints.append(points)
    #             if len(gtDontCarePolsNum)>0 :
    #                 # 遍历 每一个 gt中为"###"的 框
    #                 for dontCarePol in gtDontCarePolsNum:
    #                     dontCarePol = gtPols[dontCarePol]
    #                     intersected_area = get_intersection(dontCarePol,detPol)
    #                     pdDimensions = detPol.area()
    #                     precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
    #                     if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
    #                         detDontCarePolsNum.append( len(detPols)-1 )
    #                         break
                                
    #         evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n")
            
    #         if len(gtPols)>0 and len(detPols)>0:
    #             #Calculate IoU and precision matrixs
    #             outputShape=[len(gtPols), len(detPols)]

    #             iouMat = np.empty(outputShape)

    #             gtRectMat = np.zeros(len(gtPols), np.int8)
    #             detRectMat = np.zeros(len(detPols), np.int8)

    #             # 一张图片标签里的 第i个的 框
    #             tiouRecallMat = np.empty(outputShape)  
    #             tiouPrecisionMat = np.empty(outputShape)  

    #             # 
    #             # tiouGtRectMat = np.zeros(len(gtPols), np.int8) 
    #             # tiouDetRectMat = np.zeros(len(detPols), np.int8) 

    #             for gtNum in range(len(gtPols)):
    #                 for detNum in range(len(detPols)):
    #                     pG = gtPols[gtNum]
    #                     pD = detPols[detNum]

    #                     # IOU_gt_i
    #                     iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
    #                     # TIOU_recall_gt_i
    #                     tiouRecallMat[gtNum,detNum] = get_text_intersection_over_union_recall(pD, pG)
    #                     # TIOU_precision_i
    #                     tiouPrecisionMat[gtNum,detNum] = get_text_intersection_over_union_precision(pD, pG, gtNum, gtPols, gtDontCarePolsNum)  

    #             for gtNum in range(len(gtPols)):
    #                 for detNum in range(len(detPols)):
    #                     if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
    #                         if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:

    #                             # IOU
    #                             gtRectMat[gtNum] = 1
    #                             detRectMat[detNum] = 1

    #                             # SIOU
    #                             detMatched += 1
    #                             detMatched_iou += iouMat[gtNum,detNum]

    #                             # TIOU
    #                             detMatched_tiouGt += tiouRecallMat[gtNum,detNum] 
    #                             detMatched_tiouDt += tiouPrecisionMat[gtNum,detNum]

    #                             if  iouMat[gtNum,detNum] != tiouRecallMat[gtNum,detNum]: 
    #                                 detMatched_cutGt +=1

    #                             if  iouMat[gtNum,detNum] != tiouPrecisionMat[gtNum,detNum]: 
    #                                 detMatched_coverOtherGt +=1

    #                             pairs.append({'gt':gtNum,'det':detNum})
    #                             detMatchedNums.append(detNum)
    #                             evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

    #         if evaluationParams['CONFIDENCES']:
    #             for detNum in range(len(detPols)):
    #                 if detNum not in detDontCarePolsNum :
    #                     #we exclude the don't care detections
    #                     match = detNum in detMatchedNums

    #                     arrSampleConfidences.append(confidencesList[detNum])
    #                     arrSampleMatch.append(match)

    #                     arrGlobalConfidences.append(confidencesList[detNum])
    #                     arrGlobalMatches.append(match)

    #     # 除去 无效的框
    #     numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
    #     numDetCare = (len(detPols) - len(detDontCarePolsNum))
    #     if numGtCare == 0:
    #         recall = float(1)
    #         precision = float(0) if numDetCare >0 else float(1)
    #         sampleAP = precision
    #         iouRecall = float(1)
    #         iouPrecision = float(0) if numDetCare > 0 else float(1)
    #         tiouRecall = float(1) 
    #         tiouPrecision = float(0) if numDetCare >0 else float(1)
    #     else:
    #         recall = float(detMatched) / numGtCare
    #         precision = 0 if numDetCare==0 else float(detMatched) / numDetCare

    #         iouRecall = float(detMatched_iou) / numGtCare 
    #         iouPrecision = 0 if numDetCare==0 else float(detMatched_iou) / numDetCare 

    #         tiouRecall = float(detMatched_tiouGt) / numGtCare 
    #         tiouPrecision = 0 if numDetCare==0 else float(detMatched_tiouDt) / numDetCare 

    #         if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
    #             sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )

    #     hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)
    #     tiouHmean = 0 if (tiouPrecision + tiouRecall)==0 else 2.0 * tiouPrecision * tiouRecall / (tiouPrecision + tiouRecall)     
    #     iouHmean = 0 if (iouPrecision + iouRecall)==0 else 2.0 * iouPrecision * iouRecall / (iouPrecision + iouRecall)     

    #     # SIOU
    #     matchedSum += detMatched
    #     matchedSum_iou += detMatched_iou 

    #     # TIOU
    #     matchedSum_tiouGt += detMatched_tiouGt 
    #     matchedSum_tiouDt += detMatched_tiouDt 

    #     matchedSum_cutGt += detMatched_cutGt 
    #     matchedSum_coverOtherGt += detMatched_coverOtherGt 

    #     # 一张图片中 有效框的数量
    #     numGlobalCareGt += numGtCare
    #     numGlobalCareDet += numDetCare
        
    #     # 将结果保存为.txt
    #     if evaluationParams['PER_SAMPLE_RESULTS']:
    #         perSampleMetrics[resFile] = {
    #                                         'precision':precision,
    #                                         'recall':recall,
    #                                         'hmean':hmean,
    #                                         'iouPrecision':iouPrecision,
    #                                         'iouRecall':iouRecall,
    #                                         'iouHmean':iouHmean,
    #                                         'tiouPrecision':tiouPrecision,
    #                                         'tiouRecall':tiouRecall,
    #                                         'tiouHmean':tiouHmean,
    #                                         'pairs':pairs,
    #                                         'AP':sampleAP,
    #                                         'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
    #                                         'gtPolPoints':gtPolPoints,
    #                                         'detPolPoints':detPolPoints,
    #                                         'gtDontCare':gtDontCarePolsNum,
    #                                         'detDontCare':detDontCarePolsNum,
    #                                         'evaluationParams': evaluationParams,
    #                                         'evaluationLog': evaluationLog                                        
    #                                     }
    #     #fper_.writelines(resFile+'\t"IoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})",\t"TIoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})".\n'.format(precision, recall, hmean, tiouPrecision, tiouRecall, tiouHmean))
    #     try:
    #         totalNumGtPols += len(gtPols) 
    #         totalNumDetPols += len(detPols)
    #     except Exception as e:
    #         raise e
    
    # #fper_.close()

    # # Compute MAP and MAR
    # AP = 0
    # if evaluationParams['CONFIDENCES']:
    #     AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    # print('num_gt, num_det: ', numGlobalCareGt, totalNumDetPols)

    # # iou
    # methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    # methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    # methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    # # siou
    # methodRecall_iou = 0 if numGlobalCareGt == 0 else float(matchedSum_iou)/numGlobalCareGt 
    # methodPrecision_iou = 0 if numGlobalCareDet == 0 else float(matchedSum_iou)/numGlobalCareDet 
    # iouMethodHmean = 0 if methodRecall_iou + methodPrecision_iou==0 else 2* methodRecall_iou * methodPrecision_iou / (methodRecall_iou + methodPrecision_iou) 

    # # tiou
    # methodRecall_tiouGt = 0 if numGlobalCareGt == 0 else float(matchedSum_tiouGt)/numGlobalCareGt 
    # methodPrecision_tiouDt = 0 if numGlobalCareDet == 0 else float(matchedSum_tiouDt)/numGlobalCareDet 
    # tiouMethodHmean = 0 if methodRecall_tiouGt + methodPrecision_tiouDt==0 else 2* methodRecall_tiouGt * methodPrecision_tiouDt / (methodRecall_tiouGt + methodPrecision_tiouDt) 
    
    # methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean}
    # iouMethodMetrics = {'iouPrecision':methodPrecision_iou, 'iouRecall':methodRecall_iou,'iouHmean': iouMethodHmean }
    # tiouMethodMetrics = {'tiouPrecision':methodPrecision_tiouDt, 'tiouRecall':methodRecall_tiouGt,'tiouHmean': tiouMethodHmean }

    # et = time.time()
    # print("[INFO] Time is:{}".format(et - st))

    # # print('matchedSum: ', matchedSum, 'matchedSum_cutGt: ', matchedSum_cutGt, 'cut_Rate: ', round(matchedSum_cutGt*1.0/matchedSum, 3), 'matchedSum_coverOtherGt: ', matchedSum_coverOtherGt, 'cover_Outlier_Rate: ', round(matchedSum_coverOtherGt*1.0/matchedSum, 3))
    # print('Origin:')
    # print("recall: ", round(methodRecall,3), "precision: ", round(methodPrecision,3), "hmean: ", round(methodHmean,3))
    # print('SIoU-metric:')
    # print("iouRecall:", round(methodRecall_iou,3), "iouPrecision:", round(methodPrecision_iou,3), "iouHmean:", round(iouMethodHmean,3))
    # print('TIoU-metric:')
    # print("tiouRecall:", round(methodRecall_tiouGt,3), "tiouPrecision:", round(methodPrecision_tiouDt,3), "tiouHmean:", round(tiouMethodHmean,3))

    # resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics, 'iouMethod': iouMethodMetrics, 'tiouMethod': tiouMethodMetrics}
    
 

 #   # 多线程 - 切分

    # threads = []
    # fper_ = open('per_samle_result.txt', 'w')

    # import time
    # stt = time.time()

    # for t in range(2):
    #     t= Mythread(t, fper_, evaluationParams, gt, subm, t*250+1, (t+1)*250+1)
    #     threads.append(t)

    # for thr in threads:
    #     thr.start()

    # for thr in threads:
    #     thr.join()
    
    # numGlobalCareGt_ = 0
    # numGlobalCareDet_ = 0
    # matchedSum_tiouGt_ = 0
    # matchedSum_tiouDt_ = 0


    # for thr in threads:        
    #     numGlobalCareGt_ += thr.result_multiprocess[0]
    #     numGlobalCareDet_ += thr.result_multiprocess[1]
    #     matchedSum_tiouGt_ += thr.result_multiprocess[2]
    #     matchedSum_tiouDt_ += thr.result_multiprocess[3]

    # methodRecall_tiouGt_ = 0 if numGlobalCareGt_ == 0 else float(matchedSum_tiouGt_)/numGlobalCareGt_
    # methodPrecision_tiouDt_ = 0 if numGlobalCareDet_ == 0 else float(matchedSum_tiouDt_)/numGlobalCareDet_ 
    # tiouMethodHmean = 0 if methodRecall_tiouGt_ + methodPrecision_tiouDt_ ==0 else 2* methodRecall_tiouGt_ * methodPrecision_tiouDt_ / (methodRecall_tiouGt_ + methodPrecision_tiouDt_) 

    # ett = time.time()
    # print("[INFO] multi process Time is :", (ett - stt))
    
    # print("methodRecall_tiouGt_:", methodRecall_tiouGt_)
    # print("methodPrecision_tiouDt_:", methodPrecision_tiouDt_)
    # print("tiouMethodHmean:", tiouMethodHmean)

 #   # 多进程 - 切分
    from multiprocessing import Manager
    stt = time.time()

    # key=id  value=label
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)

    fper_ = open('per_samle_result.txt', 'w')
    with Manager() as manager:
        result_list = manager.list()

        Multi_Process = []
        num_  = num
        length = len(gt) // num_
        for P in range(num_):
            P= MyMultiProcess(P, result_list, fper_, evaluationParams, gt, subm, P*length+1, (P+1)*length+1)
            Multi_Process.append(P)

        for Pro in Multi_Process:
            Pro.start()

        for Pro in Multi_Process:
            Pro.join()
        
        numGlobalCareGt_ = 0
        numGlobalCareDet_ = 0
        matchedSum_tiouGt_ = 0
        matchedSum_tiouDt_ = 0                
        perSampleMetrics = dict()
        for res in result_list:           
            numGlobalCareGt_ += res[0]
            numGlobalCareDet_ += res[1]
            matchedSum_tiouGt_ += res[2]
            matchedSum_tiouDt_ += res[3]
            perSampleMetrics.update(res[4])

        methodRecall_tiouGt_ = 0 if numGlobalCareGt_ == 0 else float(matchedSum_tiouGt_)/numGlobalCareGt_
        methodPrecision_tiouDt_ = 0 if numGlobalCareDet_ == 0 else float(matchedSum_tiouDt_)/numGlobalCareDet_ 
        tiouMethodHmean = 0 if methodRecall_tiouGt_ + methodPrecision_tiouDt_ ==0 else 2* methodRecall_tiouGt_ * methodPrecision_tiouDt_ / (methodRecall_tiouGt_ + methodPrecision_tiouDt_) 

        ett = time.time()
        print("[INFO] 多进程 Time is :", (ett - stt))        
        print("methodRecall_tiouGt_:", methodRecall_tiouGt_)
        print("methodPrecision_tiouDt_:", methodRecall_tiouGt_)
        print("tiouMethodHmean:", tiouMethodHmean)

        tiouMethodMetrics = {'tiouPrecision':methodRecall_tiouGt_, 'tiouRecall':methodRecall_tiouGt_,'tiouHmean': tiouMethodHmean }
        resDict = {'calculated':True,'Message':'','per_sample': perSampleMetrics, 'tiouMethod': tiouMethodMetrics}

    return  resDict




################################  以下 为 必要 的 函数 ####################################

#  多线程 模式
class Mythread(threading.Thread):
    def __init__(self, threadID, fper_, evaluationParams, gt, subm, startt=None, endd=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.fper_ = fper_
        self.evaluationParams = evaluationParams
        self.gt = gt
        self.subm = subm
        self.startt = startt
        self.endd = endd
        self.result_multiprocess = None
        self.threadLock = threading.Lock()

    def run(self):
        # print ("开启线程： ", self.threadID)
        # 获取锁，用于线程同步
        # self.threadLock.acquire()
        self.result_multiprocess = calculated_multi_process(self.fper_, self.evaluationParams, self.gt, self.subm, self.startt, self.endd)
        # 释放锁，开启下一个线程
        # self.threadLock.release()

#  多进程 - 切分 模式
class MyMultiProcess(Process):
    def __init__(self, ProcessID, result_list, fper_, evaluationParams, gt, subm, start_index=None, end_index=None):
        Process.__init__(self)
        self.ProcessID = ProcessID
        self.result_list = result_list
        self.fper_ = fper_
        self.evaluationParams = evaluationParams
        self.gt = gt
        self.subm = subm
        self.start_index = start_index
        self.end_index = end_index
        self.result_multi_process = None

    def run(self):
        self.result_multi_process = calculated_multi_process(self.fper_, self.evaluationParams, self.gt, self.subm, self.start_index, self.end_index)
        self.result_list.append(self.result_multi_process)

def calculated_multi_process(fper_, evaluationParams, gt, subm, start, end):

    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """        
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0])
        resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2])
        resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4])
        resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6])
        resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg.Polygon( pointMat)
    
    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax)
        resBoxes[0,7]=int(rect.ymax)

        pointMat = resBoxes[0].reshape([2,4]).T
        
        return plg.Polygon( pointMat)
    
    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points
        
    def get_union(pD,pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)
        
    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def funcCt(x):
        if x<=0.01:
            return 1
        else:
            return 1-x
    
    def get_text_intersection_over_union_recall(pD, pG):
        '''
        Ct (cut): Area of ground truth that is not covered by detection bounding box.
        '''
        try:
            Ct = pG.area() - get_intersection(pD, pG)
            assert(Ct>=0 and Ct<=pG.area()), 'Invalid Ct value'
            assert(pG.area()>0), 'Invalid Gt'

            #     TIOU_recall =  A(Gi交Di)*f(Ct)
            return (get_intersection(pD, pG) * funcCt(Ct*1.0/pG.area())) / get_union(pD, pG)
        except Exception as e:
            return 0

    
    def funcOt(x):
        if x<=0.01:
            return 1
        else:
            return 1-x

    
    def get_text_intersection_over_union_precision(pD, pG, gtNum, gtPolys, gtDontCarePolsNum):
        '''
        Ot: Outlier gt area
        '''
        Ot = 0
        try:
            inside_pG = pD & pG
            gt_union_inside_pD = None
            gt_union_inside_pD_and_pG = None
            count_initial = 0
            for i in range(len(gtPolys)):
                if i!= gtNum and gtNum not in gtDontCarePolsNum: # ignore don't care regions
                    if not get_intersection(pD, gtPolys[i]) == 0:
                        if count_initial == 0:
                            # initial 
                            gt_union_inside_pD = gtPolys[i]
                            gt_union_inside_pD_and_pG = inside_pG & gtPolys[i]
                            count_initial = 1
                            continue
                        gt_union_inside_pD = gt_union_inside_pD | gtPolys[i]
                        inside_pG_i = inside_pG & gtPolys[i]
                        gt_union_inside_pD_and_pG = gt_union_inside_pD_and_pG | inside_pG_i

            if not gt_union_inside_pD == None:
                pD_union_with_other_gt = pD & gt_union_inside_pD
                
                #
                Ot = pD_union_with_other_gt.area() - gt_union_inside_pD_and_pG.area()
                if Ot <=1.0e-10:
                    Ot = 0
            else:
                Ot = 0
            assert(Ot>=0 and Ot<=pD.area()+0.001), 'Invalid Ot value: '+str(Ot)+' '+str(pD.area())
            assert(pD.area()>0), 'Invalid pD: '+str(pD.area())

            return (get_intersection(pD, pG) * funcOt(Ot*1.0/pD.area())) / get_union(pD, pG)
        except Exception as e:
            print(e)
            return 0


    def get_intersection(pD,pG):
        
        pInt = pD & pG

        if len(pInt) == 0:
            return 0
        return pInt.area()

    
    def get_intersection_three(pD,pG,pGi):
        pInt = pD & pG
        pInt_3 = pInt & pGi
        if len(pInt_3) == 0:
            return 0
        return pInt_3.area()
    
    def compute_ap(confList, matchList,numGtCare):
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if numGtCare>0:
                AP /= numGtCare
            
        return AP
    
    perSampleMetrics = {}
    matchedSum = 0
    matchedSum_iou = 0 

    # TIOU
    matchedSum_tiouGt = 0 
    matchedSum_tiouDt = 0 

    matchedSum_cutGt = 0 
    matchedSum_coverOtherGt = 0 

    # 一张图片中 有效框的数量
    numGlobalCareGt = 0
    numGlobalCareDet = 0

    totalNumGtPols = 0
    totalNumDetPols = 0

    # fper_ = open('per_samle_result.txt', 'w')
    for resFile in range(start, end):
        # print(type(resFile))
        resFile = str(resFile)
        
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        # print("gtFile:", gtFile)
        recall = 0
        precision = 0
        hmean = 0    
        
        detMatched = 0
        detMatched_iou = 0 
        detMatched_tiouGt = 0 
        detMatched_tiouDt = 0 
        detMatched_cutGt = 0 
        detMatched_coverOtherGt = 0 
        
        iouMat = np.empty([1,1])
        
        # Polygon : shape=[[x1,y1], [x2,y2],......]
        gtPols = []
        detPols = []
        # raw gt/pre decode point: shape= x1,y1,x2,y2,x3,y3,x4,y4
        gtPolPoints = []
        detPolPoints = []  
        
        #Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        #Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []
        
        pairs = []
        detMatchedNums = []
        
        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""
        
        # 按行 读取 gt label
        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:            
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                # 记录gt中为“###”的索引
                gtDontCarePolsNum.append( len(gtPols)-1 )
                
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")
        

        # 读取 gt 对应的 pre输出的结果
        if resFile in subm:
            
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 
            
            pointsList, confidencesList, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],False,evaluationParams['CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]
                
                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)                    
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum)>0 :
                    # 遍历 每一个 gt中为"###"的 框
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break
                                
            evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n")
            
            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols), len(detPols)]

                iouMat = np.empty(outputShape)

                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)

                # 一张图片标签里的 第i个的 框
                tiouRecallMat = np.empty(outputShape)  
                tiouPrecisionMat = np.empty(outputShape)  

                # 
                # tiouGtRectMat = np.zeros(len(gtPols), np.int8) 
                # tiouDetRectMat = np.zeros(len(detPols), np.int8) 

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]

                        # IOU_gt_i
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                        # TIOU_recall_gt_i
                        tiouRecallMat[gtNum,detNum] = get_text_intersection_over_union_recall(pD, pG)
                        # TIOU_precision_i
                        tiouPrecisionMat[gtNum,detNum] = get_text_intersection_over_union_precision(pD, pG, gtNum, gtPols, gtDontCarePolsNum)  

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:

                                # IOU
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1

                                # SIOU
                                detMatched += 1
                                detMatched_iou += iouMat[gtNum,detNum]

                                # TIOU
                                detMatched_tiouGt += tiouRecallMat[gtNum,detNum] 
                                detMatched_tiouDt += tiouPrecisionMat[gtNum,detNum]

                                if  iouMat[gtNum,detNum] != tiouRecallMat[gtNum,detNum]: 
                                    detMatched_cutGt +=1

                                if  iouMat[gtNum,detNum] != tiouPrecisionMat[gtNum,detNum]: 
                                    detMatched_coverOtherGt +=1

                                pairs.append({'gt':gtNum,'det':detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum :
                        #we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        # 除去 无效的框
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))        
        numDetCare = (len(detPols) - len(detDontCarePolsNum))

        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare >0 else float(1)
            sampleAP = precision
            iouRecall = float(1)
            iouPrecision = float(0) if numDetCare > 0 else float(1)
            tiouRecall = float(1) 
            tiouPrecision = float(0) if numDetCare >0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare==0 else float(detMatched) / numDetCare

            iouRecall = float(detMatched_iou) / numGtCare 
            iouPrecision = 0 if numDetCare==0 else float(detMatched_iou) / numDetCare 

            tiouRecall = float(detMatched_tiouGt) / numGtCare 
            tiouPrecision = 0 if numDetCare==0 else float(detMatched_tiouDt) / numDetCare 

            if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )

        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)
        tiouHmean = 0 if (tiouPrecision + tiouRecall)==0 else 2.0 * tiouPrecision * tiouRecall / (tiouPrecision + tiouRecall)     
        iouHmean = 0 if (iouPrecision + iouRecall)==0 else 2.0 * iouPrecision * iouRecall / (iouPrecision + iouRecall)     

        # SIOU
        matchedSum += detMatched
        matchedSum_iou += detMatched_iou 

        # TIOU
        matchedSum_tiouGt += detMatched_tiouGt 
        matchedSum_tiouDt += detMatched_tiouDt 

        matchedSum_cutGt += detMatched_cutGt 
        matchedSum_coverOtherGt += detMatched_coverOtherGt 

        # 一张图片中 有效框的数量
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        
        # 将结果保存为.txt
        if evaluationParams['PER_SAMPLE_RESULTS']:
            perSampleMetrics[resFile] = {
                                            'precision':precision,
                                            'recall':recall,
                                            'hmean':hmean,
                                            'iouPrecision':iouPrecision,
                                            'iouRecall':iouRecall,
                                            'iouHmean':iouHmean,
                                            'tiouPrecision':tiouPrecision,
                                            'tiouRecall':tiouRecall,
                                            'tiouHmean':tiouHmean,
                                            'pairs':pairs,
                                            'AP':sampleAP,
                                            'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
                                            'gtPolPoints':gtPolPoints,
                                            'detPolPoints':detPolPoints,
                                            'gtDontCare':gtDontCarePolsNum,
                                            'detDontCare':detDontCarePolsNum,
                                            'evaluationParams': evaluationParams,
                                            'evaluationLog': evaluationLog                                        
                                        }
        fper_.writelines(resFile+'\t"IoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})",\t"TIoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})".\n'.format(precision, recall, hmean, tiouPrecision, tiouRecall, tiouHmean))
        try:
            totalNumGtPols += len(gtPols) 
            totalNumDetPols += len(detPols)
        except Exception as e:
            raise e
    fper_.close()

    res = [numGlobalCareGt, numGlobalCareDet, matchedSum_tiouGt, matchedSum_tiouDt, perSampleMetrics]

    return res


#  生产者 - 消费者 模式
def producer_multi_process_dataloader_prepeocess(q, evaluationParams, img, gt, subm):

    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """        
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0])
        resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2])
        resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4])
        resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6])
        resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg.Polygon( pointMat)
    
    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax)
        resBoxes[0,7]=int(rect.ymax)

        pointMat = resBoxes[0].reshape([2,4]).T
        
        return plg.Polygon( pointMat)
    
    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points
        
    def get_union(pD,pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)
        
    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def funcCt(x):
        if x<=0.01:
            return 1
        else:
            return 1-x
    
    def get_text_intersection_over_union_recall(pD, pG):
        '''
        Ct (cut): Area of ground truth that is not covered by detection bounding box.
        '''
        try:
            Ct = pG.area() - get_intersection(pD, pG)
            assert(Ct>=0 and Ct<=pG.area()), 'Invalid Ct value'
            assert(pG.area()>0), 'Invalid Gt'

            #     TIOU_recall =  A(Gi交Di)*f(Ct)
            return (get_intersection(pD, pG) * funcCt(Ct*1.0/pG.area())) / get_union(pD, pG)
        except Exception as e:
            return 0

    def funcOt(x):
        if x<=0.01:
            return 1
        else:
            return 1-x

    def get_text_intersection_over_union_precision(pD, pG, gtNum, gtPolys, gtDontCarePolsNum):
        '''
        Ot: Outlier gt area
        '''
        Ot = 0
        try:
            inside_pG = pD & pG
            gt_union_inside_pD = None
            gt_union_inside_pD_and_pG = None
            count_initial = 0
            for i in range(len(gtPolys)):
                if i!= gtNum and gtNum not in gtDontCarePolsNum: # ignore don't care regions
                    if not get_intersection(pD, gtPolys[i]) == 0:
                        if count_initial == 0:
                            # initial 
                            gt_union_inside_pD = gtPolys[i]
                            gt_union_inside_pD_and_pG = inside_pG & gtPolys[i]
                            count_initial = 1
                            continue
                        gt_union_inside_pD = gt_union_inside_pD | gtPolys[i]
                        inside_pG_i = inside_pG & gtPolys[i]
                        gt_union_inside_pD_and_pG = gt_union_inside_pD_and_pG | inside_pG_i

            if not gt_union_inside_pD == None:
                pD_union_with_other_gt = pD & gt_union_inside_pD
                
                #
                Ot = pD_union_with_other_gt.area() - gt_union_inside_pD_and_pG.area()
                if Ot <=1.0e-10:
                    Ot = 0
            else:
                Ot = 0
            assert(Ot>=0 and Ot<=pD.area()+0.001), 'Invalid Ot value: '+str(Ot)+' '+str(pD.area())
            assert(pD.area()>0), 'Invalid pD: '+str(pD.area())

            return (get_intersection(pD, pG) * funcOt(Ot*1.0/pD.area())) / get_union(pD, pG)
        except Exception as e:
            print(e)
            return 0


    def get_intersection(pD,pG):
        
        pInt = pD & pG

        if len(pInt) == 0:
            return 0
        return pInt.area()

    
    def get_intersection_three(pD,pG,pGi):
        pInt = pD & pG
        pInt_3 = pInt & pGi
        if len(pInt_3) == 0:
            return 0
        return pInt_3.area()
    
    def compute_ap(confList, matchList,numGtCare):
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if numGtCare>0:
                AP /= numGtCare
            
        return AP
    
    perSampleMetrics = {}
    matchedSum = 0
    matchedSum_iou = 0 

    # TIOU
    matchedSum_tiouGt = 0 
    matchedSum_tiouDt = 0 

    matchedSum_cutGt = 0 
    matchedSum_coverOtherGt = 0 

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    totalNumGtPols = 0
    totalNumDetPols = 0

    img = rrc_evaluation_funcs.load_zip_file(p['m'], evalParams['Test_Image_ID'])

    # nparr = np.frombuffer(img['1'], dtype=np.uint8)
    # segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gt = rrc_evaluation_funcs.load_zip_file(gt,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(subm,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)

    # fper_ = open('per_samle_result.txt', 'w')
    rt = []
    for resFile in gt:
        gtt = time.time()
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])

        recall = 0
        precision = 0
        hmean = 0    
        
        detMatched = 0
        detMatched_iou = 0 
        detMatched_tiouGt = 0 
        detMatched_tiouDt = 0 
        detMatched_cutGt = 0 
        detMatched_coverOtherGt = 0 
        
        iouMat = np.empty([1,1])
        
        # Polygon : shape=[[x1,y1], [x2,y2],......]
        gtPols = []
        detPols = []
        # raw gt/pre decode point: shape= x1,y1,x2,y2,x3,y3,x4,y4
        gtPolPoints = []
        detPolPoints = []  
        
        #Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        #Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []
        
        pairs = []
        detMatchedNums = []
        
        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""
        
        # 按行 读取 gt label
        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:            
                gtPol = polygon_from_points(points)
                
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                # 记录gt中为“###”的索引
                gtDontCarePolsNum.append( len(gtPols)-1 )
                
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum)>0 else "\n")
        

        # 读取 gt 对应的 pre输出的结果
        if resFile in subm:
            
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 
            
            pointsList, confidencesList, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],False,evaluationParams['CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]
                
                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)                    
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum)>0 :
                    # 遍历 每一个 gt中为"###"的 框
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break

            
            evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum)>0 else "\n")
            
            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols), len(detPols)]

                iouMat = np.empty(outputShape)

                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)

                # 一张图片标签里的 第i个的 框
                tiouRecallMat = np.empty(outputShape)  
                tiouPrecisionMat = np.empty(outputShape)  

                # 
                # tiouGtRectMat = np.zeros(len(gtPols), np.int8) 
                # tiouDetRectMat = np.zeros(len(detPols), np.int8) 

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]

                        # IOU_gt_i
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
                        # TIOU_recall_gt_i
                        tiouRecallMat[gtNum,detNum] = get_text_intersection_over_union_recall(pD, pG)
                        # TIOU_precision_i
                        tiouPrecisionMat[gtNum,detNum] = get_text_intersection_over_union_precision(pD, pG, gtNum, gtPols, gtDontCarePolsNum)  

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:

                                # IOU
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1

                                # SIOU
                                detMatched += 1
                                detMatched_iou += iouMat[gtNum,detNum]

                                # TIOU
                                detMatched_tiouGt += tiouRecallMat[gtNum,detNum] 
                                detMatched_tiouDt += tiouPrecisionMat[gtNum,detNum]

                                if  iouMat[gtNum,detNum] != tiouRecallMat[gtNum,detNum]: 
                                    detMatched_cutGt +=1

                                if  iouMat[gtNum,detNum] != tiouPrecisionMat[gtNum,detNum]: 
                                    detMatched_coverOtherGt +=1

                                pairs.append({'gt':gtNum,'det':detNum})
                                detMatchedNums.append(detNum)
                                # evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

            # if evaluationParams['CONFIDENCES']:
            #     for detNum in range(len(detPols)):
            #         if detNum not in detDontCarePolsNum :
            #             #we exclude the don't care detections
            #             match = detNum in detMatchedNums

            #             arrSampleConfidences.append(confidencesList[detNum])
            #             arrSampleMatch.append(match)

            #             arrGlobalConfidences.append(confidencesList[detNum])
            #             arrGlobalMatches.append(match)

        # 除去 无效的框
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))        
        numDetCare = (len(detPols) - len(detDontCarePolsNum))

        # if numGtCare == 0:
        #     recall = float(1)
        #     precision = float(0) if numDetCare >0 else float(1)
        #     sampleAP = precision
        #     iouRecall = float(1)
        #     iouPrecision = float(0) if numDetCare > 0 else float(1)
        #     tiouRecall = float(1) 
        #     tiouPrecision = float(0) if numDetCare >0 else float(1)
        # else:
        #     recall = float(detMatched) / numGtCare
        #     precision = 0 if numDetCare==0 else float(detMatched) / numDetCare


        #     iouRecall = float(detMatched_iou) / numGtCare 
        #     iouPrecision = 0 if numDetCare==0 else float(detMatched_iou) / numDetCare 

        #     tiouRecall = float(detMatched_tiouGt) / numGtCare 
        #     tiouPrecision = 0 if numDetCare==0 else float(detMatched_tiouDt) / numDetCare 

        #     if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
        #         sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )

        # hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)
        # tiouHmean = 0 if (tiouPrecision + tiouRecall)==0 else 2.0 * tiouPrecision * tiouRecall / (tiouPrecision + tiouRecall)     
        # iouHmean = 0 if (iouPrecision + iouRecall)==0 else 2.0 * iouPrecision * iouRecall / (iouPrecision + iouRecall)   

        # fper_.writelines(resFile+'\t"IoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})",\t"TIoU: (P: {:.3f}. R: {:.3f}. F: {:.3f})".\n'.format(precision, recall, hmean, tiouPrecision, tiouRecall, tiouHmean))
        # fper_.close() 


    #######################################
    #            可视化
    #######################################
        # 将 bytes类型转numpy，转换成图像
        nparr = np.frombuffer(img[resFile], dtype=np.uint8)
        segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # 生成白色图像
        newImg = np.ones((segment_data.shape[0], segment_data.shape[1]//4, segment_data.shape[2]), np.uint8)
        newImg[:] = [255,255,255]

        for i in range(len(gtPols)):

            x1, y1 = list(gtPols[i][0][0])
            x2, y2 = list(gtPols[i][0][1])
            x3, y3 = list(gtPols[i][0][2])
            x4, y4 = list(gtPols[i][0][3])

            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            x3, y3, x4, y4 = int(x3),int(y3),int(x4),int(y4)

            text = transcriptionsList[i]

            cv2.line(segment_data, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)
            cv2.line(segment_data, (int(x2), int(y2)), (int(x3), int(y3)), (255, 0, 0), thickness=2)
            cv2.line(segment_data, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), thickness=2)
            cv2.line(segment_data, (int(x1), int(y1)), (int(x4), int(y4)), (255, 0, 0), thickness=2)

            cv2.putText(segment_data, str(text), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)

        for i in range(len(detPols)):

            x1, y1 = list(detPols[i][0][0])
            x2, y2 = list(detPols[i][0][1])
            x3, y3 = list(detPols[i][0][2])
            x4, y4 = list(detPols[i][0][3])

            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            x3, y3, x4, y4 = int(x3),int(y3),int(x4),int(y4)

            cv2.line(segment_data, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2)
            cv2.line(segment_data, (int(x2), int(y2)), (int(x3), int(y3)), (0, 0, 255), thickness=2)
            cv2.line(segment_data, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 255), thickness=2)
            cv2.line(segment_data, (int(x1), int(y1)), (int(x4), int(y4)), (0, 0, 255), thickness=2)

        # 拼接图像
        segment_data = np.concatenate([newImg, segment_data], axis=1)

        cv2.putText(segment_data, str('recall: {}'.format(round(recall, 4))), (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 50, 255), 1)
        cv2.putText(segment_data, str('precision: {}'.format(round(precision, 4))), (30, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 50, 255), 1)
        cv2.putText(segment_data, str('hmean: {}'.format(round(hmean, 4))), (30, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 50, 255), 1)
        cv2.putText(segment_data, str('tiouRecall:{}'.format(round(tiouRecall,4))), (30, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(segment_data, str('tiouPrecision: {}'.format(round(tiouPrecision,4))), (30, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(segment_data, str('tiouHmean: {}'.format(round(tiouHmean, 4))), (30, 300), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

        cv2.imwrite("./ic15/result_img/img_"+resFile+".jpg", segment_data)

        try:
            totalNumGtPols += len(gtPols) 
            totalNumDetPols += len(detPols)
        except Exception as e:
            raise e
        perSampleMetrics[resFile] = []
        # if evaluationParams['PER_SAMPLE_RESULTS']:
        #     perSampleMetrics[resFile] = {
        #                                     'precision':precision,
        #                                     'recall':recall,
        #                                     'hmean':hmean,
        #                                     'iouPrecision':iouPrecision,
        #                                     'iouRecall':iouRecall,
        #                                     'iouHmean':iouHmean,
        #                                     'tiouPrecision':tiouPrecision,
        #                                     'tiouRecall':tiouRecall,
        #                                     'tiouHmean':tiouHmean,
        #                                     'pairs':pairs,
        #                                     'AP':sampleAP,
        #                                     'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
        #                                     'gtPolPoints':gtPolPoints,
        #                                     'detPolPoints':detPolPoints,
        #                                     'gtDontCare':gtDontCarePolsNum,
        #                                     'detDontCare':detDontCarePolsNum,
        #                                     'evaluationParams': evaluationParams,
        #                                     'evaluationLog': evaluationLog                                        
        #                                 }  
        result_per = [detMatched, detMatched_iou, detMatched_tiouGt, detMatched_tiouDt, numGtCare, numDetCare, perSampleMetrics[resFile]]
        # print('producer :{}生产了数据：{}'.format(os.getpid(),resFile))
        q.put(result_per)
        rt.append((time.time() - gtt))
        # print("处理数据时间：",(time.time() - gtt))

        # return [detMatched, detMatched_iou, detMatched_tiouGt, detMatched_tiouDt, numGtCare, numDetCare]
    # fper_.close()
    # print("全部时间：",sum(rt), "平均时间：", sum(rt)/500)
    q.put(None)

def consumer_multi_process_calculation_result(q):

    matchedSum = 0
    matchedSum_iou = 0

    # TIOU
    matchedSum_tiouGt = 0 
    matchedSum_tiouDt = 0 

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    resFile = 1
    perSampleMetrics = {}

    # stime = time.time()

    while True:
        res=q.get() # [detMatched, detMatched_iou,detMatched_tiouGt, detMatched_tiouDt, numGtCare, numDetCare]
        # print('consumer :{} 消费了数据：{}'.format(os.getpid(),resFile))
        if res == None:
            break

        # SIOU
        matchedSum += res[0]  # detMatched
        matchedSum_iou += res[1]  # detMatched_iou 

        # TIOU
        matchedSum_tiouGt += res[2]  # detMatched_tiouGt 
        matchedSum_tiouDt += res[3]  # detMatched_tiouDt 

        numGlobalCareGt += res[4]  # numGtCare
        numGlobalCareDet += res[5]  # numDetCare

        perSampleMetrics[str(resFile)] = res[6] # perSampleMetrics[resFile]
        resFile += 1
    
    # iou
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    # siou
    methodRecall_iou = 0 if numGlobalCareGt == 0 else float(matchedSum_iou)/numGlobalCareGt 
    methodPrecision_iou = 0 if numGlobalCareDet == 0 else float(matchedSum_iou)/numGlobalCareDet 
    iouMethodHmean = 0 if methodRecall_iou + methodPrecision_iou==0 else 2* methodRecall_iou * methodPrecision_iou / (methodRecall_iou + methodPrecision_iou) 

    # tiou
    methodRecall_tiouGt = 0 if numGlobalCareGt == 0 else float(matchedSum_tiouGt)/numGlobalCareGt 
    methodPrecision_tiouDt = 0 if numGlobalCareDet == 0 else float(matchedSum_tiouDt)/numGlobalCareDet 
    tiouMethodHmean = 0 if methodRecall_tiouGt + methodPrecision_tiouDt==0 else 2* methodRecall_tiouGt * methodPrecision_tiouDt / (methodRecall_tiouGt + methodPrecision_tiouDt) 
    
    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean}
    iouMethodMetrics = {'iouPrecision':methodPrecision_iou, 'iouRecall':methodRecall_iou,'iouHmean': iouMethodHmean }
    tiouMethodMetrics = {'tiouPrecision':methodPrecision_tiouDt, 'tiouRecall':methodRecall_tiouGt,'tiouHmean': tiouMethodHmean }

    # print('matchedSum: ', matchedSum, 'matchedSum_cutGt: ', matchedSum_cutGt, 'cut_Rate: ', round(matchedSum_cutGt*1.0/matchedSum, 3), 'matchedSum_coverOtherGt: ', matchedSum_coverOtherGt, 'cover_Outlier_Rate: ', round(matchedSum_coverOtherGt*1.0/matchedSum, 3))
    print('Origin:')
    print("recall: ", round(methodRecall,3), "precision: ", round(methodPrecision,3), "hmean: ", round(methodHmean,3))
    print('SIoU-metric:')
    print("iouRecall:", round(methodRecall_iou,3), "iouPrecision:", round(methodPrecision_iou,3), "iouHmean:", round(iouMethodHmean,3))
    print('TIoU-metric:')
    print("tiouRecall:", round(methodRecall_tiouGt,3), "tiouPrecision:", round(methodPrecision_tiouDt,3), "tiouHmean:", round(tiouMethodHmean,3))

    # print("生产者 - 消费者 Time is :", (time.time() - stime))
    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics, 'iouMethod': iouMethodMetrics, 'tiouMethod': tiouMethodMetrics}
    
if __name__=='__main__':
        
    rrc_evaluation_funcs.main_evaluation(None, default_evaluation_params, validate_data, evaluate_method, num=5)


    #  生产者 - 消费者 模式
    # # p : {'g': 'ic15/gt.zip', 's': 'ic15/pixellinkch4.zip'}
    # # {'g': 'ic15/gt.zip', 's': 'ic15/pixellinkch4.zip', 'm': 'ic15/ch4_test_images.zip'}
    
    # p = dict([s[1:].split('=') for s in sys.argv[1:]])
    # evalParams = default_evaluation_params()
    # q=Queue()
    # #生产者
    # p1=Process(target=producer_multi_process_dataloader_prepeocess,args=(q, evalParams, p['m'], p['g'], p['s']))
    # #消费者
    # c1=Process(target=consumer_multi_process_calculation_result,args=(q,))
    # #开始
    # p1.start()
    # c1.start()
