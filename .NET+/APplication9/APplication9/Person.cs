using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace APplication9
{
    public class Person: INotifyPropertyChanged
    {
        public int Age { get; set; }

        private string name;

        public string Name
        {
            get { return name; }
            set {
                if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Name"));
                name = value; }
        }


        public bool isMale { get; set; }



        public event PropertyChangedEventHandler PropertyChanged;
    }

    static class PersonMgr
    {
        public static ObservableCollection<Person> Persons
        {
            get
            {
                var result = new ObservableCollection<Person>();
                for (int i = 0; i < 20; i++)
                {
                    result.Add(new Person
                    {
                        Age = 10 + i,
                        Name = "Jack"+i,
                        isMale = i % 2 == 0
                    }
                               );
                }
                return result;
            }
        }
    }
}
