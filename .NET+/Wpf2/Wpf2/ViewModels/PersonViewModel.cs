using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using Wpf2.Models;

namespace Wpf2.ViewModels
{
    public class PersonViewModel
    {
        public Person Person { get;private set; }

        public Brush Color 
        { 
        get {
            if (Person.Age < 18) 
                return Brushes.Honeydew; 
            else return Brushes.Navy; 
            } 
         
        }


        public PersonViewModel(Person person)
        {
            this.Person = person;
        }
    }
}
