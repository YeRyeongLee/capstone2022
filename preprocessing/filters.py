filter_schizo = ['diagnosed me with schizophrenia', 
                 'diagnosed me with psychosis',
                 'diagnosed with schizophrenia',
                 'diagnosed with psychosis',
                 'diagnosed with schizoaffective disorder',
                 'i am schizophrenic',
                 'i have schizophrenia',
                 'i have schizoaffective disorder',
                 'i have psychosis',
                 'my schizophrenia']

filter_ADHD = ['i have adhd',
               'diagnosed with adhd',
               'diagnosed me with adhd'
               'living with adhd',
               'my adhd']

filter_bipolar = ['my bipolar',
                  'hypomanic',
                  'hypomania',
                  'manic',
                  'mania',
                  'depressive',
                  'i have bipolar',
                  'diagnosed bipolar',
                  'i am bipolar',
                  'being bipolar',
                  'i have bp2',
                  'diagnosed with bipolar']

filter_depress = ['got diagnosed',
                  'diagnosed with depression',
                  'i have depression',
                  'suffering from depression',
                  'my depression',
                  "i'm depressed",
                  'been depressed',
                  'dealing with depression',
                  'diagnosed me with anxiety']

filter_Anxiety = ['dealing with anxiety',
                  'my anxiety',
                  'have anxiety',
                  'suffering from anxiety',
                  "i'm anxious",
                  'got diagnosed',
                  'diagnosed me with anxiety',
                  'diagnosed with social anxiety',
                  'diagnosed with anxiety',
                  'diagnosed with sad', 
                  'diagnosed with gad']

filter_dict = {
    'schizophrenia': filter_schizo,
    'ADHD': filter_ADHD, 
    'bipolar': filter_bipolar,
    'depression': filter_depress,
    'Anxiety': filter_Anxiety
}