# Files that are unknown to svn --> delete_list.txt
svn st | grep ^? | awk '{print $2}' > delete_list.txt

# Create compile test directory
mkdir compile_test
mv compile_test ..

# Copy all elements from the current directory
for element in `ls .`; do
  # directory
  if [ -d $element ]; then
    ln -s `pwd`/$element ../compile_test/$element
  fi

  # file
  if [ -f $element ]; then
    cp $element ../compile_test/
  fi
done

# Remove files that are unknown to svn
rm -r `cat delete_list.txt | awk '{print "../compile_test/"$1}'`
rm ./delete_list.txt
