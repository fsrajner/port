using GalaSoft.MvvmLight;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wpf2.Models
{
    public class Person: ObservableObject
    {
        public int Age { get; set; }

        private string name;

        public string Name
        {
            get { return name; }
            set {
                Set(() => Name, ref name, value);
                }
        }

    }
}
