'''
Created on Dec 26, 2011

@author: Weiyi Sun

Provides some alternative ways to evaluates system output Tlinks against gold standard Tlinks:

The temporal closure in this script is completed using SputLink (SputLink citation?) 

This script assumes gold standard EVENTs/TIMEX3s in both gold standard and system output.
If that is not the case, please make sure that the EVENT/TIMEX3 ids in the gold standard match
the ids in the system output, or the result will be wrong.

- usage:
  $ python tlinkEvaluation.py [--oc] [--oo] [--cc] goldstandard_xml_filename system_output_xml_filename
  
  --oc: Original against Closure [default]:
        -Precision: the total number of system output Tlinks that can be verified in the gold standard closure
                    divided by the total number of system output Tlinks
        -Recall: the total number gold standard output Tlinks that can be verified in the system closure
                 divided by the total number of gold standard output Tlinks
  --oo: Original against Original:
        -Precision: the total number of system output Tlinks that can be verified in the gold standard output
                    divided by the total number of system output Tlinks
        -Recall: the total number gold standard output Tlinks that can be verified in the system output
                 divided by the total number of gold standard output Tlinks  
  --cc: Closure against Closure:
        -Precision: the total number of system closure Tlinks that can be verified in the gold standard closure
                    divided by the total number of system output Tlinks
        -Recall: the total number gold standard closure Tlinks that can be verified in the system closure
                 divided by the total number of gold standard output Tlinks  
'''
import sys

if sys.version_info<(2,7):
    print("Error: This evaluation script requires Python 2.7 or higher")
else:
    import argparse
    import os
    import re
    import time
    import subprocess
    import stat
    
    
    header = """<fragment>
    <TEXT>
    """
    
    footer = """
    </fragment>"""
    
    _DEBUG = True
    #_DEBUG = False
    
    def open_file(fname):
        if os.path.exists(fname):
            f = open(fname)
            return f
        else:
            outerror("No such file: %s" % fname)
            return None
        
    def outerror(text):
        #sys.stderr.write(text + "\n")
        raise Exception(text)
    
    def attr_by_line(tlinkline):
        """
        Args:
          line - str: MAE TLINK tag line,
                      e.g. <TLINK id="TL70" fromID="E28" fromText="her erythema"
                       toID="E26" toText="erythema on her left leg" type="OVERLAP" />
        """
        re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+'
        m = re.search(re_exp, tlinkline)
        if m:
            id, fromid, fromtext, toid, totext, attr_type = m.groups()
        else:
            raise Exception("Malformed EVENT tag: %s" % (tlinkline))
        attr_type=attr_type.replace('SIMULTANEOUS','OVERLAP')
        return id, fromid, toid, attr_type.upper()
    
    def attr_by_closure(cline): #event to event
        """
        Args:
          line - str: Sputlink closure tag line,
                      e.g. <TLINK origin=" i" toEID="E49" fromEID="E48" relType="SIMULTANEOUS"/>
        """
        re_exp = 'origin=\"([^"]*)\"\s+to[ET]ID=\"([^"]*)\"\s+from[ET]ID=\"([^"]*)\"\s+relType=\"([^"]*)\"+\/>'
        re_exp = 'origin=\"([^"]*)\"\s+relType=\"([^"]*)\"\s+from[ET]ID=\"([^"]*)\"\s+to[ET]ID=\"([^"]*)\"+'
        m = re.search(re_exp, cline)
        if m:
            id, toid, fromid, attr_type = m.groups()
        else:
            raise Exception("Malformed EtoE tag: %s" % (cline))
        attr_type=attr_type.replace('SIMULTANEOUS','OVERLAP')
        return fromid, toid, attr_type
    
    def attr_by_closure2(cline): #event to timex
        """
        Args:
          line - str: Sputlink closure tag line,
                      e.g. <TLINK fromTID="T12" origin=" i" toEID="E32" relType="SIMULTANEOUS"/>
        """
        re_exp = 'from[TE]ID=\"([^"]*)\"\s+origin=\"([^"]*)\"\s+to[ET]ID=\"([^"]*)\"\s+relType=\"([^"]*)\"+'
        re_exp = 'from[TE]ID=\"([^"]*)\"\s+origin=\"([^"]*)\"\s+relType=\"([^"]*)\"\s+to[ET]ID=\"([^"]*)\"'
        m = re.search(re_exp, cline)
        if m:
            fromid, id, toid, attr_type = m.groups()
        else:
            raise Exception("Malformed EtoT tag: %s" % (cline))
        attr_type=attr_type.replace('SIMULTANEOUS','OVERLAP')
        return fromid, toid, attr_type
    
    def get_tlinks(text_fname):
        '''
        Args:
            text_fname: file name of the MAE xml file
        
        Output:
            a tlinks tuple of all the tlinks in the file 
        '''
        tf=open(text_fname[:-5])
        lines = tf.readlines()
        tlinks=[]
        for line in lines:  
            if re.search('<TLINK',line):
                tlink_tuple=attr_by_line(line)
                tlinks.append(tlink_tuple)
        return tlinks
    
    def get_tlinks_closure(text_fname):
        '''
        Args:
            text_fname: file name of the tlink closure xml file
        
        Output:
            a tlinks tuple of all the valid tlink output in the closure     
        '''
        temp_dir=os.path.join(os.getcwd(),'sputlink','closure_temp')
        filename=re.split('[/\\\\]',text_fname)[-1]
        cfilename=os.path.join(temp_dir,filename+'.'+str(os.stat(text_fname[:-5]).st_mtime)+'.closure.xml')
        clines = open(cfilename).readlines()
        orig_tlinks=get_tlinks(text_fname)
        closed_tlinks=[]
        existing_tlinks=[]
        for tlink in orig_tlinks:
            closed_tlinks.append(tlink)
            existing_tlinks.append(tlink[1:])
        for cline in clines:
            id=''
            if not re.search('<TLINK origin|relType=\"\"|relType=\"INCLUDES\"',cline): #T to E or T to T
                if re.search('<TLINK fromTID',cline): 
                    if re.search('origin=\"closure\"',cline):
                        id="closure"
                    elif re.search('origin=\" i\"',cline):
                        id="inverse"
                    elif re.search('origin=\"\"',cline):
                        id="default"
                    if id!='':
                        closed_tlink_tuple=attr_by_closure2(cline)
                        if closed_tlink_tuple not in existing_tlinks:
                            fromid, toid, attr_type=closed_tlink_tuple
                            closed_tlinks.append([id,fromid, toid, attr_type])
                            existing_tlinks.append([closed_tlink_tuple])
            elif not re.search('relType=\"\"|relType=\"INCLUDES\"',cline): # E to E or E to T
                if re.search('origin=\"closure\"',cline):
                    id="closure"
                elif re.search('origin=\" i\"',cline):
                    id="inverse"
                elif re.search('origin=\"\"',cline):
                    id="default"
                if id!='':
                    closed_tlink_tuple=attr_by_closure(cline)
                    if closed_tlink_tuple not in existing_tlinks:
                        fromid, toid, attr_type=closed_tlink_tuple
                        closed_tlinks.append([id,fromid, toid, attr_type])
                        existing_tlinks.append([closed_tlink_tuple])     

        return closed_tlinks
    
    def compare_tlinks(text_fname1, text_fname2, dic, option='OrigVsClosure'):
        '''
        This function verifies whether the TLinks in text_fname1 can be found in text_fname2
        using the evaluation method in 'option' arg:
        
        Args:
            text_fname1:    filename of the first xml file 
            text_fname2:    filename of the second xml file
            dic:            a dictionary that maps extent id in the first file to
                            the corresponding extent id in the second file
            option:         OrigVsClosure | ClosureVsClosure | OrigVsOrig
        
        Output:
            totalcomlinks:    Total number of comparable TLINKs (tlinks whose extents 
                              were annotated by both xml files)
            totalmatch:       Total number of matched tlinks 
        '''
        if option == 'OrigVsClosure':
            tlinks_tuple1=get_tlinks_closure(text_fname1)
            tlinks_tuple2=get_tlinks(text_fname2)
        elif option=='ClosureVsClosure':
            tlinks_tuple1=get_tlinks_closure(text_fname1)
            tlinks_tuple2=get_tlinks_closure(text_fname2)
            print(tlinks_tuple1)
            print(tlinks_tuple2)
        elif option=='OrigVsOrig':
            tlinks_tuple1=get_tlinks(text_fname1)
            tlinks_tuple2=get_tlinks(text_fname2)
        
        totalcomlinks=len(tlinks_tuple2)
        totalmatch=0
        dic['Admission']='Admission'
        dic['Discharge']='Discharge'
        for tlinks2 in tlinks_tuple2:
            if len(tlinks2)==4:
                linkid2=tlinks2[0]
                fromid2=tlinks2[1]
                toid2=tlinks2[2]
                if fromid2!='' and toid2!='' and linkid2.find('R')==-1 and fromid2.find('S')==-1 and toid2.find('S')==-1:
                    if len(dic)==2:
                        fromid_tuple = [fromid2]
                        toid_tuple = [toid2]
                    else:
                        try:
                            fromid_tuple = dic[fromid2].split('#@#')
                        except KeyError:
                            fromid_tuple=['']
                            print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % fromid2)
                        try:
                            toid_tuple = dic[toid2].split('#@#')
                        except KeyError:
                            toid_tuple=['']
                            print("\n%s" %  "Error: Unknown EVENT or TIMEX id in TLINK: %s" % toid2)
                    type2=tlinks2[3]
                    match=0
                    fromid=fromid_tuple[0]
                    toid=toid_tuple[0]
                    
                    for fromid in fromid_tuple:
                        for toid in toid_tuple:
                            if fromid!="" and toid!="":
                                for tlinks1 in tlinks_tuple1:
                                    if len(tlinks1)==4:
                                        if fromid==tlinks1[1] and toid==tlinks1[2]:
                                            type1=tlinks1[3]                                   
                                            if type2==type1:
                                                match=1
                                                break
                    totalmatch+=match
        return totalcomlinks, totalmatch
    
    def tlinkClosurePreprocess(text_fname):
        '''
        process the xml file and output it in a format that can be
        processed in SputLink. The output file will be placed in the
        SputLink directory. 
            e.g. input file dir1/dir2/file
                 output file sputlink/dir1/dir2/file
        
        Args:
            text_fname: name of the MAE xml file to be processed
        '''
        tf = open_file(text_fname[:-5])
        lines=tf.readlines()
        temp_dir=os.path.join(os.getcwd(),'sputlink','closure_temp')
        filename=re.split('[/\\\\]',text_fname)[-1]
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        nfname=os.path.join(temp_dir,filename+'.pcd.xml')
        nf=open(nfname, 'w')
        nf.write(header)
        count=3
        #fix of handling the first TLINK
        firstTlinkFlag=0
        for i in range(3, len(lines)):      
            if not re.search("]]><",lines[i]): 
                #nf.write(lines[i])
                count+=1
            else:
                nf.write("</TEXT>\n")
                break
        count+=2
        for i in range(count, len(lines)):      
            if re.search("<EVENT id",lines[i]): 
                lines[i]=lines[i].replace(" type=", " eventtype=")
                pre,post=lines[i].split(" id=")
                outline=pre+" eid="+post
                outline=outline.replace(' />','></EVENT>')
                while re.search('\s[&|\'|<|>]\s', outline):
                    outline=re.sub('\s[&|\'|<|>]\s',' ', outline)
                nf.write(outline)     
            elif re.search("<TIMEX3 id",lines[i]):  
                lines[i]=lines[i].replace(" type=", " timextype=")
                pre,post=lines[i].split(" id=")
                outline=pre+" tid="+post
                outline=outline.replace(' />','></TIMEX3>')
                while re.search('\s[&|\'|<|>]\s', outline):
                    outline=re.sub('\s[&|\'|<|>]\s',' ', outline)
                nf.write(outline)     
            elif re.search("<SECTIME",lines[i]):
                lines[i]=lines[i].replace(" type=", " sectype=")   
                nf.write(lines[i])  
            elif re.search("<TLINK",lines[i]):
                if firstTlinkFlag==0:
                    nf.write('<TLINK></TLINK>\n')
                    firstTlinkFlag=1
                re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+'
                m = re.search(re_exp, lines[i])
                if m:
                    id, fromid, fromtext, toid, totext, attr_type = m.groups()
                else:
                    raise Exception("Malformed EVENT tag: %s" % (lines[i]))
                attr_type=attr_type.replace('OVERLAP','SIMULTANEOUS')
                while re.search('[&|\'|<|>]', fromtext):
                    fromtext=re.sub('[&|\'|<|>]',' ', fromtext)
                while re.search('[&|\'|<|>]', totext):
                    totext=re.sub('[&|\'|<|>]',' ', totext)
                if fromid!='' and toid!='' and attr_type!='':
                    outline='<TLINK lid=\"%s\"' % id
                    if re.search('E',fromid):
                        outline += ' fromEID=\"%s\" fromText=\"%s\"' % (fromid,fromtext)
                    else:
                        outline += ' fromTID=\"%s\" fromText=\"%s\"' % (fromid,fromtext)
                    if re.search('E',toid):
                        outline += ' toEID=\"%s\" toText=\"%s\" type=\"%s\" ></TLINK>' % (toid, totext,attr_type)
                    else:
                        outline += ' toTID=\"%s\" toText=\"%s\" type=\"%s\" ></TLINK>' % (toid,totext,attr_type)
                    nf.write(outline+'\n')         
        nf.write(footer)
    
    def tlinkEvaluation(gold_fname, system_fname, option, goldDic={}, sysDic={}):
        
        gold_fname=gold_fname+'.gold'
        system_fname=system_fname+'.syst'
        tlinkClosurePreprocess(gold_fname)
        tlinkClosurePreprocess(system_fname)

        precLinkCount, precMatchCount, recLinkCount, recMatchCount= [0, 0, 0, 0]
        if option=='OrigVsOrig':
    
            precLinkCount, precMatchCount= compare_tlinks(gold_fname, system_fname, sysDic, option)
            recLinkCount, recMatchCount = compare_tlinks(system_fname, gold_fname, goldDic, option)
            
            if precLinkCount>0:
                precision=float(precMatchCount)/precLinkCount
            if recLinkCount>0:
                recall=float(recMatchCount)/recLinkCount
    
        else:
            temp_dir=os.path.join(os.getcwd(),'sputlink','closure_temp')
            gfilename=re.split('[/\\\\]',gold_fname)[-1]
            sfilename=re.split('[/\\\\]',system_fname)[-1]
            if not os.path.isfile(os.path.join(temp_dir,gfilename+'.'+str(os.stat(gold_fname[:-5]).st_mtime)+'.closure.xml')):
                root=os.getcwd()
                absfile=os.path.abspath(gold_fname[:-5])
                path=os.path.join(root,"sputlink")
                os.chdir(path)
                nf=open('sputlink.temp','w')
                print("\nSputLink is computing the temporal closure... Please give it a moment...")
                pcdfname=os.path.join('closure_temp',gfilename+'.pcd.xml')
                closurefname=os.path.join('closure_temp',gfilename+'.'+str(os.stat(absfile).st_mtime)+'.closure.xml')
                subprocess.call(['perl','merge.pl',pcdfname,closurefname],stdout=nf,stderr=nf)
                nf.close()
                os.remove('sputlink.temp')
    
                os.chdir(root)
                
            if not os.path.isfile(os.path.join(temp_dir,sfilename+'.'+str(os.stat(system_fname[:-5]).st_mtime)+'.closure.xml')):
                root=os.getcwd()
                absfile=os.path.abspath(system_fname[:-5])
                path=os.path.join(root,"sputlink")
                os.chdir(path)
                nf=open('sputlink.temp','w')
                print("\nSputLink is computing the temporal closure... Please give it a moment...")
                pcdfname=os.path.join('closure_temp',sfilename+'.pcd.xml')
                closurefname=os.path.join('closure_temp',sfilename+'.'+str(os.stat(absfile).st_mtime)+'.closure.xml')
                
                subprocess.call(['perl','merge.pl',pcdfname,closurefname],stdout=nf,stderr=nf)
                nf.close()
                os.remove('sputlink.temp')
                os.chdir(root)
            if  os.path.isfile(os.path.join(temp_dir,gfilename+'.'+str(os.stat(gold_fname[:-5]).st_mtime)+'.closure.xml')) \
            and os.path.isfile(os.path.join(temp_dir,sfilename+'.'+str(os.stat(system_fname[:-5]).st_mtime)+'.closure.xml')):
            
                precLinkCount, precMatchCount = compare_tlinks(gold_fname, system_fname, sysDic, option)
                recLinkCount, recMatchCount = compare_tlinks(system_fname, gold_fname, goldDic, option)
                if precLinkCount>0:
                    precision=float(precMatchCount)/precLinkCount
                if recLinkCount>0:
                    recall=float(recMatchCount)/recLinkCount
        kill  
        return precLinkCount, recLinkCount, precMatchCount,  recMatchCount
        
    if __name__ == '__main__':
        usage= "%prog [options] [goldstandard-file] [systemOutput-file]" + __doc__
        parser = argparse.ArgumentParser(description='Evaluate system output TLINKs against gold standard TLINKs.')
        parser.add_argument('gold_file', type=str, nargs=1, \
                            help='gold standard xml file')
        parser.add_argument('system_file', type=str, nargs=1,
                         help='system output xml file')
        parser.add_argument('--cc', dest='evaluation_option', action='store_const',\
                          const='cc', default='oc', help='select different attr_types of tlink evaluation: oc - original against closure; cc - closure against closure; oo - original against original (default: oc)')
        parser.add_argument('--oc', dest='evaluation_option', action='store_const',\
                          const='oc', default='oc', help='select different attr_types of tlink evaluation: oc - original against closure; cc - closure against closure; oo - original against original (default: oc)')
        parser.add_argument('--oo', dest='evaluation_option', action='store_const',\
                          const='oo', default='oc', help='select different attr_types of tlink evaluation: oc - original against closure; cc - closure against closure; oo - original against original (default: oc)')
          
        args = parser.parse_args()
        # run on a single file
        gold=args.gold_file[0]
        system=args.system_file[0]
        if os.path.isfile(gold) and os.path.isfile(system):
            
            if args.evaluation_option=='oo':
                precLinkCount, recLinkCount, precMatchCount,  recMatchCount=tlinkEvaluation(gold, system,'OrigVsOrig')
            elif args.evaluation_option=='cc':
                precLinkCount, recLinkCount, precMatchCount,  recMatchCount=tlinkEvaluation(gold, system,'ClosureVsClosure')
            elif args.evaluation_option=='oc':
                precLinkCount, recLinkCount, precMatchCount,  recMatchCount=tlinkEvaluation(gold, system,'OrigVsClosure')

            if precLinkCount>0:
                precision=float(precMatchCount)/precLinkCount
            else:
                precision=0.0
            if recLinkCount>0:
                recall=float(recMatchCount)/recLinkCount
            else:
                recall=0.0
            if (precLinkCount+recLinkCount)>0:                                       
                averagePR=(precMatchCount+recMatchCount)*1.0/(precLinkCount+recLinkCount)
            else:
                averagePR=0.0
            if (precision+recall)>0:
                fScore=2*(precision*recall)/(precision+recall)
            else:
                fScore=0.0
            print("""
            Total number of comparable TLINKs: 
               Gold Standard : \t\t"""+str(recMatchCount)+"""
               System Output : \t\t"""+str(precMatchCount)+"""
            --------------
            Recall : \t\t\t"""+'%.4f'%(recall)+"""
            Precision: \t\t\t""" + '%.4f'%(precision)+"""
            Average P&R : \t\t"""+'%.4f'%(averagePR)+"""
            F measure : \t\t"""+'%.4f'%(fScore)+'\n')

            print("WARNING: Running TLINK evaluation by itself assumes gold standard EVENTs/TIMEX3s in both gold standard and system output. If that is not the case, please make sure that the EVENT/TIMEX3 ids in the gold standard match the ids in the system output, or the result will be wrong.")

        else:
            print("Error: Please input exactly 2 arguments: gold_standard_filename, system_file_name. Use i2b2Evaluation.py for evaluating two directories")
