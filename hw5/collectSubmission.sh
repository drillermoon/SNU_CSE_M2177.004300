files="GANs-TensorFlow.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required notebook $file not found."
        exit 0
    fi
done

rm -f hw5_results.zip
zip -r hw5_results.zip . -x "*.git*" "hw5_results*" "gan_outputs_tf.png"  "*cs231n/datasets*" "*.ipynb_checkpoints*" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" 
