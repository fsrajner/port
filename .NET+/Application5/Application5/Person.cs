using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application5
{
    public class Person : INotifyPropertyChanged
    {
        private int age;

        public int Age
        {
            get { return age; }
            set
            {
                if (value == age) return;
                if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("age"));
                age = value; }
        }


        public string Name { get; set; }


        public event PropertyChangedEventHandler PropertyChanged;
    }

    public class PersonList: BindingList<Person>
    {

    }
}
